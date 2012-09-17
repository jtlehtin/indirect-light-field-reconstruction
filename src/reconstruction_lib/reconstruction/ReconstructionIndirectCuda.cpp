//-------------------------------------------------------------------------------------------------
// CUDA reconstruction
//-------------------------------------------------------------------------------------------------

#pragma warning(disable:4127)		// conditional expression is constant
#include "ReconstructionIndirect.hpp"
#include "gpu/CudaCompiler.hpp"
#include "ReconstructionIndirectCudaKernels.hpp"
#include "base/Timer.hpp"
#include <cfloat>
#include <cstdio>

using namespace FW;

//-------------------------------------------------------------------------------------------------

class CudaReconstructionInd
{
public:
									CudaReconstructionInd	(float aoLength = 0.f, bool useBandwidthInformation=false, bool selectNearestSample=false);
									~CudaReconstructionInd	(void);

	void							reconstructGPU			(int outputSpp, int nr, const Vec2i& size,
															 Image& resultImage,
															 const Array<CudaPBRTRay>& rays,
															 const Array<CudaReceiverInd>& recv,
															 const Array<CudaNodeInd>& nodes,
															 const Array<CudaTNodeInd>& tnodes,
															 const Array<CudaSampleInd>& samples,
															 const Array<CudaTSampleInd>& tsamples,
															 const Array<Vec2f>& sobol,
															 bool normalizeImage);

	void							shrinkGPU				(const Array<CudaShrinkRayInd>& rays,
															 const Array<CudaTNodeInd>& tnodes,
															 const Array<CudaTSampleInd>& tsamples,
															 Array<float>& sampleRadii);

private:
	Vec4f							reconstructCPUSingle	(int smp);

private:
    CudaCompiler					m_compiler;
};

//-------------------------------------------------------------------------------------------------

CudaReconstructionInd::CudaReconstructionInd(float aoLength, bool useBandwidthInformation, bool selectNearestSample)
{
    m_compiler.setSourceFile("src/reconstruction_lib/reconstruction/ReconstructionIndirectCudaKernels.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/framework");
    m_compiler.define("SM_ARCH", sprintf("%d", CudaModule::getComputeCapability()));
	if(aoLength>0)
		m_compiler.define("AMBIENT_OCCLUSION", sprintf("%f", aoLength));
	if(useBandwidthInformation)
		m_compiler.define("USE_BANDWIDTH_INFORMATION");
	if(selectNearestSample)
		m_compiler.define("SELECT_NEAREST_SAMPLE");
}

CudaReconstructionInd::~CudaReconstructionInd(void)
{
	// empty
}

//-------------------------------------------------------------------------------------------------

static float totaltime_gpu = 0.f;
static float totaltime_both = 0.f;

void CudaReconstructionInd::reconstructGPU(int outputSpp, int nr, const Vec2i& size,
										   Image& resultImage,
										   const Array<CudaPBRTRay>& in_rays,
										   const Array<CudaReceiverInd>& in_recv,
										   const Array<CudaNodeInd>& in_nodes,
										   const Array<CudaTNodeInd>& in_tnodes,
										   const Array<CudaSampleInd>& in_samples,
										   const Array<CudaTSampleInd>& in_tsamples,
										   const Array<Vec2f>& in_sobol,
										   bool normalizeImage)
{
	// set up kernel
	CudaModule* module       = m_compiler.compile();
	CudaKernel  filterKernel = module->getKernel("filterKernel");

	// clear CPU result image
	resultImage.clear(Vec4f(0));

	Buffer recv    (in_recv.getPtr(), in_recv.getNumBytes());
	Buffer nodes   (in_nodes.getPtr(), in_nodes.getNumBytes());
	Buffer tnodes  (in_tnodes.getPtr(), in_tnodes.getNumBytes());
	Buffer samples (in_samples.getPtr(), in_samples.getNumBytes());
	Buffer tsamples(in_tsamples.getPtr(), in_tsamples.getNumBytes());
	module->setTexRef("t_nodes",   tnodes,   CU_AD_FORMAT_FLOAT, 4);
	module->setTexRef("t_samples", tsamples, CU_AD_FORMAT_FLOAT, 4);
	Buffer sobol   (in_sobol.getPtr(), in_sobol.getNumBytes());

	FW::printf("nodes:     %d\n", in_nodes.getNumBytes());
	FW::printf("tnodes:    %d\n", in_tnodes.getNumBytes());
	FW::printf("samples:   %d\n", in_samples.getNumBytes());
	FW::printf("tsamples:  %d\n", in_tsamples.getNumBytes());

	if (in_rays.getSize())
		FW::printf("Reconstruction on GPU using PBRT ray dump\n");

	U32 N = in_rays.getSize() ? in_rays.getSize() : (in_recv.getSize() * nr);
	U32 maxBatchSize = 32 << 10;
	U32 numBatches = (N+maxBatchSize-1)/maxBatchSize;
	FW::printf("batches: %d\n", numBatches);
	float t = 0.f;

	Buffer resultImg(0, size.x * size.y * sizeof(Vec4f));
	resultImg.clear(0); // GPU result image

	IndirectKernelInput& in = *(IndirectKernelInput*)module->getGlobal("c_IndirectKernelInput").getMutablePtr();

	Buffer rays(0, maxBatchSize * sizeof(CudaPBRTRay)); // PBRT ray dumps
	in.pbrtRays     = rays.getCudaPtr();
	in.pbrtRayCount = in_rays.getSize();
	in.recv         = recv.getCudaPtr();
	in.nodes        = nodes.getCudaPtr();
	in.samples      = samples.getCudaPtr();
	in.sobol        = sobol.getCudaPtr();
	in.resultImg    = resultImg.getMutableCudaPtr();
	in.size         = make_int2(size.x, size.y);
	in.outputSpp    = outputSpp;
	in.nr           = nr;

	*((U32*)module->getGlobal("g_overflowCount").getMutablePtr()) = 0;
	*((U32*)module->getGlobal("g_emptyCount").getMutablePtr()) = 0;

	Timer timer;
	timer.start();
	for (U32 batch = 0; batch < numBatches; batch++)
	{
		U32 first = batch * maxBatchSize;
		U32 num   = min(N - first, maxBatchSize);

		FW::printf("%d / %d (%.1f %%) ..\r", batch, numBatches, 100.f * first / N);
		fflush(stdout);

		// copy PBRT rays
		if (in_rays.getSize())
			rays.setRange(0, in_rays.getPtr(first), num * sizeof(CudaPBRTRay));

		IndirectKernelInput& in = *(IndirectKernelInput*)module->getGlobal("c_IndirectKernelInput").getMutablePtr();
		in.firstRay = first;
		in.numRays  = num;
		module->updateGlobals();

		t += filterKernel.launchTimed((int)num, Vec2i(32, 4));
	}

	FW::printf("%d / %d (%.1f %%)          \n", numBatches, numBatches, 100.f);
	fflush(stdout);

	int numOverflow = *((U32*)module->getGlobal("g_overflowCount").getPtr());
	int numEmpty = *((U32*)module->getGlobal("g_emptyCount").getPtr());

	// copy result image to CPU
	Vec4f* rptr = (Vec4f*)resultImg.getPtr();
	for (int y=0; y < size.y; y++)
	for (int x=0; x < size.x; x++)
		resultImage.setVec4f(Vec2i(x, y), rptr[x+size.x*y]);

	// normalize image
	if (normalizeImage)
	{
		for (int y=0; y < size.y; y++)
		for (int x=0; x < size.x; x++)
		{
			Vec4f c = resultImage.getVec4f(Vec2i(x, y));
			if (c.w == 0.f)
				c = Vec4f(0,0,0,1);
			else
				c *= (1.f/c.w);
			c.w = 1.f;
			resultImage.setVec4f(Vec2i(x, y), c);
		}
	}

	timer.end();

	float tboth = timer.getTotal();

	totaltime_gpu  += t;
	totaltime_both += tboth;

	// stats
	FW::printf("kernel launch time:   %.3f s\n", t);
	FW::printf("total CPU+GPU time:   %.3f s\n", tboth);
	FW::printf("total rays:           %d\n", N);
	FW::printf("rays with overflow:   %d\n", numOverflow);
	FW::printf("rays with 0 samples:  %d (%.2f %%)\n", numEmpty, 100.f * numEmpty / N);
}

//-------------------------------------------------------------------------------------------------

void CudaReconstructionInd::shrinkGPU(const Array<CudaShrinkRayInd>& in_rays,
									  const Array<CudaTNodeInd>& in_tnodes,
									  const Array<CudaTSampleInd>& in_tsamples,
									  Array<float>& in_sampleRadii)
{
	// set up kernel
	CudaModule* module       = m_compiler.compile();
	CudaKernel  shrinkKernel = module->getKernel("shrinkKernel");

	Buffer rays    (in_rays.getPtr(), in_rays.getNumBytes());
	Buffer tnodes  (in_tnodes.getPtr(), in_tnodes.getNumBytes());
	Buffer tsamples(in_tsamples.getPtr(), in_tsamples.getNumBytes());
	Buffer radii;
	radii.wrapCPU(in_sampleRadii.getPtr(), in_sampleRadii.getNumBytes());
	module->setTexRef("t_nodes",   tnodes,   CU_AD_FORMAT_FLOAT, 4);
	module->setTexRef("t_samples", tsamples, CU_AD_FORMAT_FLOAT, 4);

	U32 N = in_rays.getSize();
	U32 maxBatchSize = 256 << 10;
	U32 numBatches = (N+maxBatchSize-1)/maxBatchSize;
	FW::printf("batches: %d\n", numBatches);
	float t = 0.f;

	IndirectKernelInput& in = *(IndirectKernelInput*)module->getGlobal("c_IndirectKernelInput").getMutablePtr();
	in.rays  = rays.getCudaPtr();
	in.radii = radii.getMutableCudaPtr();

	Timer timer;
	timer.start();
	for (U32 batch = 0; batch < numBatches; batch++)
	{
		U32 first = batch * maxBatchSize;
		U32 num   = min(N - first, maxBatchSize);

		FW::printf("%d / %d (%.1f %%) ..\r", batch, numBatches, 100.f * first / N);
		fflush(stdout);

		IndirectKernelInput& in = *(IndirectKernelInput*)module->getGlobal("c_IndirectKernelInput").getMutablePtr();
		in.firstRay = first;
		in.numRays  = num;
		module->updateGlobals();

		t += shrinkKernel.launchTimed((int)num, Vec2i(32, 4));
	}

	FW::printf("%d / %d (%.1f %%)          \n", numBatches, numBatches, 100.f);
	fflush(stdout);

	timer.end();

	// copy radii back to CPU
	radii.getPtr();

	// stats
	FW::printf("kernel launch time:   %.3f s\n", t);
	FW::printf("total CPU+GPU time:   %.3f s\n", timer.getTotal());
	FW::printf("shrink rays:          %d\n", N);
}

//-------------------------------------------------------------------------------------------------

inline bool isSecondaryHitpointValid(const UVTSampleBuffer& sbuf,int x,int y,int i)	{ Vec3f sec_hitpoint=sbuf.getSampleExtra<Vec3f>(CID_SEC_HITPOINT, x,y,i); return sec_hitpoint.max() < 1e10f; }
inline bool isSecondaryOriginValid  (const UVTSampleBuffer& sbuf,int x,int y,int i)	{ Vec3f sec_origin=sbuf.getSampleExtra<Vec3f>(CID_SEC_ORIGIN, x,y,i);   return sec_origin.max() < 1e10f; }

void ReconstructIndirect::filterImageCuda(Image& resultImage)
{
	profilePush("Filter");

	int cid_pri_normal = m_sbuf->getChannelID(CID_PRI_NORMAL_NAME  );
	int cid_albedo     = m_sbuf->getChannelID(CID_ALBEDO_NAME      );
	int cid_sec_origin = m_sbuf->getChannelID(CID_SEC_ORIGIN_NAME  );
	int cid_sec_albedo = m_sbuf->getChannelID(CID_SEC_ALBEDO_NAME  );
	int cid_sec_direct = m_sbuf->getChannelID(CID_SEC_DIRECT_NAME  );

	bool canGather = (cid_sec_albedo >= 0 && cid_sec_direct >= 0);

	bool selectNearestSample     = m_selectNearestSample;
	bool useBandwidthInformation = m_useBandwidthInformation;

	resultImage.clear(0xff884422);
	Vec2i size = resultImage.getSize();

	// copy samples
	Array<CudaSampleInd>  samples (0, m_samples.getSize());
	Array<CudaTSampleInd> tsamples(0, m_samples.getSize());
	for (int i=0; i < m_samples.getSize(); i++)
	{
		Sample&	        ismp = m_samples[i];
		CudaSampleInd&  osmp = samples[i];
		CudaTSampleInd& tsmp = tsamples[i];

		osmp.pos    = ismp.sec_hitpoint;
		osmp.normal = ismp.sec_normal;
		osmp.orig   = ismp.sec_origin;
		osmp.color  = ismp.color;
		osmp.size   = ismp.radius;
		osmp.plen   = ismp.sec_origin.length(); // camera is at origin, so this is primary ray length
		osmp.bw     = m_sbuf->getSampleW(ismp.origIndex.x, ismp.origIndex.y, ismp.origIndex.z);

		tsmp.posSize    = Vec4f(osmp.pos, osmp.size);
		tsmp.normalPlen = Vec4f(osmp.normal, osmp.plen);
//		tsmp.color      = Vec4f(osmp.color, 1.f);
	}

	// copy tree
	Array<CudaNodeInd> nodes;
	Array<CudaTNodeInd> tnodes;
	Array<int> nodeRemap(0, m_hierarchy.getSize());
	for (int i=0; i < m_hierarchy.getSize(); i++)
	{
		Node& inode = m_hierarchy[i];
		if (inode.isLeaf())
		{
			// tag last sample into sample arrays
			CudaSampleInd&  osmp = samples[inode.s1-1];
			CudaTSampleInd& tsmp = tsamples[inode.s1-1];
			F32 size = min(-fabs(osmp.size), -FW_F32_MIN);
			osmp.size = tsmp.posSize.w = size;
		} else
		{
			nodeRemap[i] = nodes.getSize();

			CudaNodeInd&  onode = nodes.add();
			CudaTNodeInd& tnode = tnodes.add();

			// for internal nodes, store child node indices
			onode.idx0 = inode.child0;
			onode.idx1 = inode.child1;

			// ensure that (idx0 > idx1) to detect as internal node
			if (onode.idx0 < onode.idx1)
				swap(onode.idx0, onode.idx1);

			// store children's bounding boxes
			Vec3f bbmin0 = m_hierarchy[onode.idx0].bbmin;
			Vec3f bbmax0 = m_hierarchy[onode.idx0].bbmax;
			Vec3f bbmin1 = m_hierarchy[onode.idx1].bbmin;
			Vec3f bbmax1 = m_hierarchy[onode.idx1].bbmax;

			onode.bbmin[0] = m_hierarchy[onode.idx0].bbmin;
			onode.bbmax[0] = m_hierarchy[onode.idx0].bbmax;
			onode.bbmin[1] = m_hierarchy[onode.idx1].bbmin;
			onode.bbmax[1] = m_hierarchy[onode.idx1].bbmax;

			if (m_hierarchy[onode.idx0].isLeaf()) onode.idx0 = ~m_hierarchy[onode.idx0].s0;
			if (m_hierarchy[onode.idx1].isLeaf()) onode.idx1 = ~m_hierarchy[onode.idx1].s0;

			tnode.hdr  = Vec4f(bitsToFloat(onode.idx0), bitsToFloat(onode.idx1), 0.f, 0.f);
			tnode.n0xy = Vec4f(bbmin0.x, bbmax0.x, bbmin0.y, bbmax0.y);
			tnode.n1xy = Vec4f(bbmin1.x, bbmax1.x, bbmin1.y, bbmax1.y);
			tnode.nz   = Vec4f(bbmin0.z, bbmax0.z, bbmin1.z, bbmax1.z);
		}
	}

	// remap child indices
	for (int i=0; i < nodes.getSize(); i++)
	{
		CudaNodeInd&  onode = nodes[i];
		CudaTNodeInd& tnode = tnodes[i];
		if (onode.idx0 >= 0) onode.idx0 = nodeRemap[onode.idx0];
		if (onode.idx1 >= 0) onode.idx1 = nodeRemap[onode.idx1];
		tnode.hdr.x = bitsToFloat(onode.idx0);
		tnode.hdr.y = bitsToFloat(onode.idx1);
	}


	// construct receiver array, take note of stuff going to invalid pixels
	Array<Vec2f> validCount(0, size.x * size.y);
	for (int i=0; i < size.x * size.y; i++)
		validCount[i] = 0.f;

	Array<CudaReceiverInd> receivers;
	int n  = m_sbuf->getNumSamples() / SUBSAMPLE_SBUF;
	int nr = m_numReconstructionRays / n;
	for (int y=0; y < size.y; y++)
	for (int x=0; x < size.x; x++)
	for (int i=0; i < n; i++)
	{
		Vec3f origin = m_sbuf->getSampleExtra<Vec3f>(cid_sec_origin, x, y, i);	// shoot secondary from here
		Vec3f normal = m_sbuf->getSampleExtra<Vec3f>(cid_pri_normal, x, y, i);	// orientation of hemisphere
		Vec3f albedo = m_sbuf->getSampleExtra<Vec3f>(cid_albedo,     x, y, i);	// albedo (needed once incident light has been computed)

		validCount[x + size.x * y].x += 1.f; // count
		if (origin.max() >= 1e10f)
			continue; // invalid primary hit
		validCount[x + size.x * y].y += 1.f; // valid as well

		CudaReceiverInd& r = receivers.add();
		r.pixel  = x + size.x * y;
		r.pos    = origin;
		r.normal = normal;
		r.albedo = albedo;
	}

	// experimental: construct receiver array out of samples
	Array<CudaReceiverInd> recvSamples(0, samples.getSize());
	int snr = 16; // number of gather samples
	if (canGather)
	{
		for (int i=0; i < samples.getSize(); i++)
		{
			CudaSampleInd&   s = samples[i];
			CudaReceiverInd& r = recvSamples[i];
			r.pixel  = i;
			r.pos    = s.pos;
			r.normal = s.normal;
			r.albedo = m_samples[i].sec_albedo;
		}
	}

	// construct the "sobol" table
	Array<Vec2f> sobolTbl(0, m_numReconstructionRays);
	for (int i=0; i < n;  i++) // number of origins per pixel
	for (int j=0; j < nr; j++) // number of reconstruction rays per origin
	{
		int idx0 = nr*i + j;
   		sobolTbl[idx0].x = sobol(1, idx0);
   		sobolTbl[idx0].y = sobol(4, idx0);
	}

	// construct the "sobol" table for gather rays
	Array<Vec2f> gatherSobolTbl(0, snr);
	for (int i=0; i < snr; i++)
	{
		gatherSobolTbl[i].x = hammersley(i, snr);
		gatherSobolTbl[i].y = larcherPillichshammer(i);
	}

	if(m_rayDumpFileName.getLength())
	{
		// multipass PBRT ray dump reconstruction
		FILE* fp = NULL;
		fopen_s(&fp, m_rayDumpFileName.getPtr(), "rb");
		if(!fp)
			fail((m_rayDumpFileName + String(" not found!")).getPtr());

		FW::printf("MULTIPASS PBRT RAY DUMP RENDER\n");

		_fseeki64(fp, 0, SEEK_END);
		const S64 fileSize = _ftelli64(fp);
		printf(" file size = %I64d bytes\n", fileSize);
		_fseeki64(fp, 0, SEEK_SET);
		const int numPBRTRays = (int)(fileSize / sizeof(PBRTReconstructionRay));
		printf(" number of rays = %d\n", numPBRTRays);
		printf(" sizeof(Ray) = %d\n", sizeof(PBRTReconstructionRay));

		// clear output image
		resultImage.clear(Vec4f(0));

		// clear stats
		totaltime_gpu = 0.f;
		totaltime_both = 0.f;

		int batchSize = 8 << 20; // in rays
		int numBatches = (numPBRTRays + batchSize - 1) / batchSize;
		for (int first = 0, batch = 0; first < numPBRTRays; first += batchSize, batch++)
		{
			int num = min(batchSize, numPBRTRays - first);

			printf("ray batch %d / %d \n", batch+1, numBatches);

			// load a set of rays and sort them
			Array<PBRTReconstructionRay> subRays(0, num);
			fread(subRays.getPtr(), sizeof(PBRTReconstructionRay), num, fp);
			FW_SORT_ARRAY_MULTICORE(subRays, ReconstructIndirect::PBRTReconstructionRay, ((int)floor(a.xy[1])*4096+(int)floor(a.xy[0]) < ((int)floor(b.xy[1])*4096+(int)floor(b.xy[0]))));

			// construct local ray array
			Array<CudaPBRTRay> pbrtRays;
			for (int i=0; i < subRays.getSize(); i++)
			{
				PBRTReconstructionRay& iray = subRays[i];

				int x = (int)iray.xy[0];
				int y = (int)iray.xy[1];
				if (x < 0 || y < 0 || x >= size.x || y >= size.y )
					continue;

				CudaPBRTRay& oray = pbrtRays.add();
				oray.pixel  = x + size.x * y;
				oray.o      = iray.o;
				oray.d      = iray.d;
				oray.weight = iray.weight;
			}

			// can happen if weird culling
			if (!pbrtRays.getSize())
				continue;

			// run reconstruction into a temporary image, no normalization in between
			receivers.reset(0);
			Image res(size, ImageFormat::RGBA_Vec4f);
			CudaReconstructionInd cr(m_aoLength,useBandwidthInformation,selectNearestSample);
			cr.reconstructGPU(m_numReconstructionRays, nr, size, res, pbrtRays, receivers, nodes, tnodes, samples, tsamples, sobolTbl, false);

			// accumulate
			for (int y=0; y < size.y; y++)
			for (int x=0; x < size.x; x++)
			{
				Vec2i pos(x, y);
				resultImage.setVec4f(pos, resultImage.getVec4f(pos) + res.getVec4f(pos));
			}
		}

		// print total timings
		FW::printf("TOTAL GPU time:      %.3f s\n", totaltime_gpu);
		FW::printf("TOTAL GPU+CPU time:  %.3f s\n", totaltime_both);

		// find divisor for result image
		float wmax = 0.f;
		for (int y=0; y < size.y; y++)
		for (int x=0; x < size.x; x++)
		{
			Vec4f c = resultImage.getVec4f(Vec2i(x, y));
			wmax = max(wmax, c.w);
		}

		FW::printf("max spp: %.2f, normalizing with this\n", wmax);

		// normalize result image
		for (int y=0; y < size.y; y++)
		for (int x=0; x < size.x; x++)
		{
			Vec4f c = resultImage.getVec4f(Vec2i(x, y));
			if (c.w == 0.f)
				c = Vec4f(0,0,0,1);
			else
				c *= (1.f/wmax);
			c.w = 1.f;
			resultImage.setVec4f(Vec2i(x, y), c);
		}

		// done
		fclose(fp);
	}
	else
	{
		// construct PBRT dump ray array if any
		Array<CudaPBRTRay> pbrtRays;
		for (int i=0; i < m_PBRTReconstructionRays.getSize(); i++)
		{
			PBRTReconstructionRay& iray = m_PBRTReconstructionRays[i];

			int x = (int)iray.xy[0];
			int y = (int)iray.xy[1];
			if (x < 0 || y < 0 || x >= size.x || y >= size.y)
				continue;

			CudaPBRTRay& oray = pbrtRays.add();
			oray.pixel  = x + size.x * y;
			oray.o      = iray.o;
			oray.d      = iray.d;
			oray.weight = iray.weight;
		}

		// experimental: gather from samples to receiver samples
		int gatherPasses = 0;
		for (int pass=0; pass < gatherPasses; pass++)
		{
			if (!canGather)
			{
				FW::printf("WARNING: Cannot perform gather pass because no secondary hit albedo/direct channel in sample buffer!\n");
				continue;
			}
			if (pbrtRays.getSize())
			{
				FW::printf("WARNING: Gather pass disabled because running from PBRT ray dump\n");
				break;
			}

			FW::printf("Starting gather pass %d\n", pass);

			// run in multiple parts in case recvSamples is too big
			Image gres(Vec2i(recvSamples.getSize(), 1), ImageFormat::RGBA_Vec4f);
			int maxBufferSize = 1 << 20; // in recv samples
			int first = 0;
			while (first < recvSamples.getSize())
			{
				int num = min(maxBufferSize, recvSamples.getSize() - first);
				Array<CudaReceiverInd> subRecv(0, num);
				for (int i=0; i < num; i++)
				{
					subRecv[i] = recvSamples[first + i];
					subRecv[i].pixel = i;
				}

				Image subres(Vec2i(num, 1), ImageFormat::RGBA_Vec4f);
				CudaReconstructionInd crg(m_aoLength,useBandwidthInformation,selectNearestSample);
				crg.reconstructGPU(snr, snr, subres.getSize(), subres, pbrtRays, subRecv, nodes, tnodes, samples, tsamples, gatherSobolTbl, true);

				for (int i=0; i < num; i++)
					gres.setVec4f(Vec2i(first + i, 0), subres.getVec4f(Vec2i(i, 0)));

				first += num;
			}
			for (int i=0; i < recvSamples.getSize(); i++)
				samples[i].color = gres.getVec4f(Vec2i(i, 0)).getXYZ() + m_samples[i].sec_direct;
		}

		FW::printf("Starting image reconstruction\n");

		CudaReconstructionInd cr(m_aoLength,useBandwidthInformation,selectNearestSample);
		cr.reconstructGPU(m_numReconstructionRays, nr, size, resultImage, pbrtRays, receivers, nodes, tnodes, samples, tsamples, sobolTbl, true);

		// fill out the invalid stuff
		for (int i=0; i < size.x * size.y; i++)
		{
			Vec2i pos(i % size.x, i / size.x);
			Vec4f c = resultImage.getVec4f(pos);
			float r = validCount[i].y / validCount[i].x; // ratio of valid pixels
			Vec4f ic(0, 0, 0, 1); // invalid color is black
			c = r * c + (1.f - r) * ic;
			resultImage.setVec4f(pos, c);
		}
	}

	printf("CUDA reconstruction done.\n");
	profilePop();
}

void ReconstructIndirect::shrinkCuda(void)
{
	profilePush("Shrinking");

	// copy samples
	Array<CudaTSampleInd> tsamples(0, m_samples.getSize());
	for (int i=0; i < m_samples.getSize(); i++)
	{
		Sample&	        ismp = m_samples[i];
		CudaTSampleInd& tsmp = tsamples[i];
		tsmp.posSize    = Vec4f(ismp.sec_hitpoint, ismp.radius);
		tsmp.normalPlen = Vec4f(ismp.sec_normal, ismp.sec_origin.length());
	}

	// copy tree
	Array<CudaNodeInd> nodes;
	Array<CudaTNodeInd> tnodes;
	Array<int> nodeRemap(0, m_hierarchy.getSize());
	for (int i=0; i < m_hierarchy.getSize(); i++)
	{
		Node& inode = m_hierarchy[i];
		if (inode.isLeaf())
		{
			// tag last sample into sample arrays
			CudaTSampleInd& tsmp = tsamples[inode.s1-1];
			F32 size = min(-fabs(tsmp.posSize.w), -FW_F32_MIN);
			tsmp.posSize.w = size;
		} else
		{
			nodeRemap[i] = nodes.getSize();

			CudaNodeInd&  onode = nodes.add();
			CudaTNodeInd& tnode = tnodes.add();

			// for internal nodes, store child node indices
			onode.idx0 = inode.child0;
			onode.idx1 = inode.child1;

			// ensure that (idx0 > idx1) to detect as internal node
			if (onode.idx0 < onode.idx1)
				swap(onode.idx0, onode.idx1);

			// store children's bounding boxes
			Vec3f bbmin0 = m_hierarchy[onode.idx0].bbmin;
			Vec3f bbmax0 = m_hierarchy[onode.idx0].bbmax;
			Vec3f bbmin1 = m_hierarchy[onode.idx1].bbmin;
			Vec3f bbmax1 = m_hierarchy[onode.idx1].bbmax;

			onode.bbmin[0] = m_hierarchy[onode.idx0].bbmin;
			onode.bbmax[0] = m_hierarchy[onode.idx0].bbmax;
			onode.bbmin[1] = m_hierarchy[onode.idx1].bbmin;
			onode.bbmax[1] = m_hierarchy[onode.idx1].bbmax;

			if (m_hierarchy[onode.idx0].isLeaf()) onode.idx0 = ~m_hierarchy[onode.idx0].s0;
			if (m_hierarchy[onode.idx1].isLeaf()) onode.idx1 = ~m_hierarchy[onode.idx1].s0;

			tnode.hdr  = Vec4f(bitsToFloat(onode.idx0), bitsToFloat(onode.idx1), 0.f, 0.f);
			tnode.n0xy = Vec4f(bbmin0.x, bbmax0.x, bbmin0.y, bbmax0.y);
			tnode.n1xy = Vec4f(bbmin1.x, bbmax1.x, bbmin1.y, bbmax1.y);
			tnode.nz   = Vec4f(bbmin0.z, bbmax0.z, bbmin1.z, bbmax1.z);
		}
	}

	// remap child indices
	for (int i=0; i < nodes.getSize(); i++)
	{
		CudaNodeInd&  onode = nodes[i];
		CudaTNodeInd& tnode = tnodes[i];
		if (onode.idx0 >= 0) onode.idx0 = nodeRemap[onode.idx0];
		if (onode.idx1 >= 0) onode.idx1 = nodeRemap[onode.idx1];
		tnode.hdr.x = bitsToFloat(onode.idx0);
		tnode.hdr.y = bitsToFloat(onode.idx1);
	}

	// construct ray array
	Array<CudaShrinkRayInd> rays(0, m_samples.getSize());
	for (int i=0; i < m_samples.getSize(); i++)
	{
		Sample&	          ismp = m_samples[i];
		CudaShrinkRayInd& ray  = rays[i];
		ray.origin   = ismp.sec_origin;
		ray.endpoint = ismp.sec_hitpoint;
	}

	// construct sample radius array
	Array<float> radii(0, m_samples.getSize());
	for (int i=0; i < m_samples.getSize(); i++)
		radii[i] = m_samples[i].radius;

	// do the shrink
	CudaReconstructionInd cr;
	cr.shrinkGPU(rays, tnodes, tsamples, radii);

	// copy sample radii back
	int numShrunk = 0;
	for (int i=0; i < m_samples.getSize(); i++)
	{
		if (m_samples[i].radius != radii[i])
			numShrunk++;

		m_samples[i].radius = radii[i];
	}

	FW::printf("splats shrunk:        %d\n", numShrunk);

	printf("CUDA shrink done.\n");
	profilePop();
}
