/*
 *  Copyright (c) 2009-2012, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
README:
	* GPU reconstruction recommended for indirect/AO, Glossy. Much faster. CPU perhaps preferred for understanding the code. The two are mostly identical.
	* CPU reconstruction supports an optional scissor rectangle. This can be very useful for trying things out in a finite time.
	* CPU path supports motion-BVH by default. This cost +10% compared to the runs in the paper (used to be a #define).
	* Glossy:
		* CUDA supports large ray dumps with a streaming algorith.
		* CPU supports only smaller in-memory buffers.
		* Ray dumps too large. We distribute .exe + scene file + .bat that generates the ray dump.
	* Fileformat. The files use multiple versions (2.0-2.2), explain one of them.
	* interpolationWeight() implements something that the paper doesn't explain.
	* App uses CUDA for gamma correction. If CUDA is not installed, the USE_CUDA define should be disabled.
		* TODO: make sure the project compiles and runs without CUDA.
*/

#pragma warning(disable:4127)		// conditional expression is constant
#include "ReconstructionIndirect.hpp"
#include "base/BinaryHeap.hpp"

namespace FW
{

//-----------------------------------------------------------------------
// Entry points
//-----------------------------------------------------------------------
	
void Reconstruction::reconstructIndirect(const UVTSampleBuffer& sbuf, int numReconstructionRays, Image& image, Image* debugImage, Vec4i scissor)
{
	profileStart();
	ReconstructIndirect ri(sbuf,numReconstructionRays,"",0,true,false,false,scissor);
	ri.filterImage(image,debugImage);
	profileEnd();
}

void Reconstruction::reconstructIndirectCuda(const UVTSampleBuffer& sbuf, int numReconstructionRays, Image& image)
{
	profileStart();
	ReconstructIndirect ri(sbuf,numReconstructionRays,"",0,true,true,false);
	printf("Filtering on GPU...\n");
	ri.filterImageCuda(image);
	profileEnd();
}

void Reconstruction::reconstructAO(const UVTSampleBuffer& sbuf, int numReconstructionRays, float aoLength, Image& image, Image* debugImage, Vec4i scissor)
{
	profileStart();
	ReconstructIndirect ri(sbuf,numReconstructionRays,"",aoLength,true,false,false,scissor);
	ri.filterImage(image,debugImage);
	profileEnd();
}

void Reconstruction::reconstructAOCuda(const UVTSampleBuffer& sbuf, int numReconstructionRays, float aoLength, Image& image)
{
	profileStart();
	ReconstructIndirect ri(sbuf,numReconstructionRays,"",aoLength,true,true,false);
	printf("Filtering on GPU...\n");
	ri.filterImageCuda(image);
	profileEnd();
}

void Reconstruction::reconstructGlossy(const UVTSampleBuffer& sbuf, String rayDumpFileName, Image& image, Image* debugImage, Vec4i scissor)
{
	profileStart();
	ReconstructIndirect ri(sbuf,0,rayDumpFileName,0,true,false,false,scissor);
	ri.filterImage(image,debugImage);
	profileEnd();
}

void Reconstruction::reconstructGlossyCuda(const UVTSampleBuffer& sbuf, String rayDumpFileName, Image& image)
{
	profileStart();
	ReconstructIndirect ri(sbuf,0,rayDumpFileName,0,true,true,false);
	printf("Filtering on GPU...\n");
	ri.filterImageCuda(image);
	profileEnd();
}

void Reconstruction::reconstructDofMotion(const UVTSampleBuffer& sbuf, int numReconstructionRays, Image& image, Image* debugImage, Vec4i scissor)
{
	profileStart();
	ReconstructIndirect ri(sbuf,numReconstructionRays,"",0,true,false,true,scissor);
	ri.filterImage(image,debugImage);
	profileEnd();
}

//-----------------------------------------------------------------------

inline bool isSecondaryHitpointValid(const UVTSampleBuffer& sbuf,int x,int y,int i)	{ Vec3f sec_hitpoint=sbuf.getSampleExtra<Vec3f>(CID_SEC_HITPOINT, x,y,i); return sec_hitpoint.max() < 1e10f; }
inline bool isSecondaryOriginValid  (const UVTSampleBuffer& sbuf,int x,int y,int i)	{ Vec3f sec_hitpoint=sbuf.getSampleExtra<Vec3f>(CID_SEC_ORIGIN, x,y,i);   return sec_hitpoint.max() < 1e10f; }

bool ReconstructIndirect::Sample::s_motionEnabled;
bool ReconstructIndirect::Node::s_motionEnabled;

ReconstructIndirect::ReconstructIndirect(const UVTSampleBuffer& sbuf, int numReconstructionRays, String rayDumpFileName, float aoLength, bool print, bool enableCUDA, bool enableMotion, Vec4i scissor)
{
	ReconstructIndirect::Sample::s_motionEnabled = enableMotion;
	ReconstructIndirect::Node::s_motionEnabled = enableMotion;

	CID_PRI_MV       = sbuf.getChannelID(CID_PRI_MV_NAME      );
	CID_PRI_NORMAL   = sbuf.getChannelID(CID_PRI_NORMAL_NAME  );
	CID_ALBEDO       = sbuf.getChannelID(CID_ALBEDO_NAME      );
	CID_SEC_ORIGIN   = sbuf.getChannelID(CID_SEC_ORIGIN_NAME  );
	CID_SEC_HITPOINT = sbuf.getChannelID(CID_SEC_HITPOINT_NAME);
	CID_SEC_MV       = sbuf.getChannelID(CID_SEC_MV_NAME      );
	CID_SEC_NORMAL   = sbuf.getChannelID(CID_SEC_NORMAL_NAME  );
	CID_DIRECT       = sbuf.getChannelID(CID_DIRECT_NAME      );
	CID_SEC_ALBEDO   = sbuf.getChannelID(CID_SEC_ALBEDO_NAME  );
	CID_SEC_DIRECT   = sbuf.getChannelID(CID_SEC_DIRECT_NAME  );

	const int w = sbuf.getWidth ();
	const int h = sbuf.getHeight();
	const int n = sbuf.getNumSamples()/SUBSAMPLE_SBUF;

	m_sbuf = &sbuf;
	m_numReconstructionRays = numReconstructionRays;
	m_samples.reset( w*h*n );		// allocate sample array (filled by buildRecursive)
	m_aoLength = aoLength;
	m_rayDumpFileName = rayDumpFileName;
	m_selectNearestSample = rayDumpFileName.getLength()>0;
	m_useBandwidthInformation = rayDumpFileName.getLength()>0;
	m_useDofMotionReconstruction = enableMotion;

	if(scissor==Vec4i(0))			// for partial image computations
		m_scissor = Vec4i(-1,-1,w,h);
	else
		m_scissor = Vec4i(max(-1,scissor[0]-1),max(-1,scissor[1]-1), min(w,scissor[2]),min(h,scissor[3]));

	//----------------------------------------------------------------------
	// Glossy: Read PBRT's ray dump (doesn't support streaming of large files -- CUDA does)
	//----------------------------------------------------------------------

	if(!enableCUDA && m_rayDumpFileName.getLength())
	{
		FILE* fp = NULL;
		fopen_s(&fp,m_rayDumpFileName.getPtr(),"rb");
		if(!fp)
			fail( (m_rayDumpFileName + String(" not found!")).getPtr() );

		printf("Importing PBRT rays (%s)\n", m_rayDumpFileName.getPtr());

		_fseeki64(fp,0,SEEK_END);
		const S64 fileSize = _ftelli64(fp);
		printf(" file size = %I64d bytes\n", fileSize);
		_fseeki64(fp,0,SEEK_SET);
		const int numPBRTRays = (int)(fileSize/sizeof(ReconstructIndirect::PBRTReconstructionRay));
		printf(" number of rays = %d\n", numPBRTRays);
		printf(" sizeof(Ray) = %d\n", sizeof(ReconstructIndirect::PBRTReconstructionRay));
		m_PBRTReconstructionRays.reset(numPBRTRays);
		fread( m_PBRTReconstructionRays.getPtr(), sizeof(ReconstructIndirect::PBRTReconstructionRay), numPBRTRays, fp);
		fclose(fp);

		printf("Sorting PBRT rays\n");

		FW_SORT_ARRAY_MULTICORE(m_PBRTReconstructionRays,ReconstructIndirect::PBRTReconstructionRay, ((int)floor(a.xy[1])*4096+(int)floor(a.xy[0]) < ((int)floor(b.xy[1])*4096+(int)floor(b.xy[0]))));

		printf("Pre-processing PBRT rays\n");

		Vec2f brectMin( FW_F32_MAX);
		Vec2f brectMax(-FW_F32_MAX);
		for(int i=0;i<m_PBRTReconstructionRays.getSize();i++)
		{
			const ReconstructIndirect::PBRTReconstructionRay& ray = m_PBRTReconstructionRays[i];
			brectMin = min(brectMin, ray.xy);
			brectMax = max(brectMax, ray.xy);
		}

		printf(" active rectangle: (%f,%f) - (%f,%f)\n", brectMin.x,brectMin.y, brectMax.x,brectMax.y);

		const int h = sbuf.getHeight();
		m_PBRTReconstructionRaysScanlineStart.reset(h);
		for(int y=0;y<h;y++)
			m_PBRTReconstructionRaysScanlineStart[y] = -1;

		//brectMax -= brectMin;		// crop rect
		for(int i=0;i<m_PBRTReconstructionRays.getSize();i++)
		{
			ReconstructIndirect::PBRTReconstructionRay& ray = m_PBRTReconstructionRays[i];
			//ray.xy -= brectMin;	/// crop rect
			const int scanline = (int)floor(ray.xy[1]);
			if(m_PBRTReconstructionRaysScanlineStart[scanline] == -1 && scanline>=0 && scanline<h)
				m_PBRTReconstructionRaysScanlineStart[scanline] = i;
		}
		printf(" rays cover (%d,%d) pixels\n", (int)floor(brectMax.x)+1,(int)floor(brectMax.y)+1);
	}

	//----------------------------------------------------------------------
	// Scan bounding box
	//----------------------------------------------------------------------

	profilePush("Scan bbox");
	int numInvalid = 0;
	Vec3f bbmin( FW_F32_MAX);
	Vec3f bbmax(-FW_F32_MAX);
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	for(int i=0;i<n;i++)
	{
		if(isSecondaryHitpointValid(sbuf,x,y,i))
		{
			const Vec3f p   = sbuf.getSampleExtra<Vec3f>(CID_SEC_HITPOINT, x,y,i);
			const Vec3f mv  = sbuf.getSampleExtra<Vec3f>(CID_SEC_MV, x,y,i);
			const float t   = sbuf.getSampleT(x,y,i);
			const Vec3f pt0 = p - t*mv; // @ t=0
			bbmin = min(bbmin,pt0);
			bbmax = max(bbmax,pt0);
		}
		else
			numInvalid++;
	}
	if(numInvalid && print)
		printf("%d samples were invalid\n", numInvalid);
	
	//if(print) printf("bbmin (%f,%f,%f) bbmax (%f,%f,%f)\n", bbmin.x,bbmin.y,bbmin.z, bbmax.x,bbmax.y,bbmax.z);
	profilePop();

	//----------------------------------------------------------------------
	// Generate morton codes
	//----------------------------------------------------------------------

	profilePush("Morton");
	Array<SortEntry> codes;
	codes.setCapacity(w*h*n);

	const int SCALE = (1<<NBITS)-1;	// [0,2^NBITS-1] per dimension
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	for(int i=0;i<n;i++)
	{
		if(isSecondaryHitpointValid(sbuf,x,y,i))
		{
			const Vec3f p   = sbuf.getSampleExtra<Vec3f>(CID_SEC_HITPOINT, x,y,i);
			const Vec3f mv  = sbuf.getSampleExtra<Vec3f>(CID_SEC_MV, x,y,i);
			const float t   = sbuf.getSampleT(x,y,i);
			      Vec3f pt0 = p - t*mv; // @ t=0
			pt0 = (pt0-bbmin) / (bbmax-bbmin) * SCALE;	// [0,SCALE]
			SortEntry& se = codes.add();
			se.code = morton( U32(pt0.x),U32(pt0.y),U32(pt0.z) );
			se.idx  = Vec3i(x,y,i);
		}
	}
	profilePop();

	//----------------------------------------------------------------------
	// Sort
	//----------------------------------------------------------------------

	profilePush("Sort");
	FW_SORT_ARRAY_MULTICORE(codes, SortEntry, a.code < b.code);	// increasing
	profilePop();

	//----------------------------------------------------------------------
	// Build a tree (resembles Kontkanen's streaming octree builder)
	//----------------------------------------------------------------------

	profilePush("Build");
	int sampleIdx = 0;
	m_totalNumSamples   = 0;
	m_totalNumLeafNodes = 0;
	m_totalNumMixedLeaf = 0;
	m_hierarchy.add();			// reserve space for root node!
	const Node* rootNode = buildRecursive(codes, sampleIdx, MAX_LEAF_SIZE, 0,3*NBITS);
	m_hierarchy[0] = *rootNode;
	profilePop();

	if(m_totalNumMixedLeaf)
		printf("%d (%.2f%%) leaf had mixed content\n", m_totalNumMixedLeaf, 100.f*m_totalNumMixedLeaf/m_totalNumLeafNodes);

	//----------------------------------------------------------------------
	// Validate densities
	//----------------------------------------------------------------------

	const float origTreeArea = getHierarchyArea(ROOT);

	profilePush("KNN");
	MulticoreLauncher launcher1;
	Array<DensityTask> tasks;
	tasks.reset(NUM_DENSITY_TASKS);
	for(int i=0;i<NUM_DENSITY_TASKS;i++)
	{
		DensityTask& task = tasks[i];
		task.init(this);
		launcher1.push(DensityTask::compute, &task, i,1);
	}
	if(print)	launcher1.popAll("Computing densities");
	else		launcher1.popAll();
	profilePop();

	U64 numSteps = 0;
	U64 numLeaf  = 0;
	U64 numIter  = 0;
	double averageVMF = 0;
	for(int i=0;i<NUM_DENSITY_TASKS;i++)
	{
		DensityTask& task = tasks[i];
		numSteps += task.m_numSteps;
		numLeaf  += task.m_numLeaf;
		numIter  += task.m_numIter;
		averageVMF += task.m_averageVMF;
	}
	if(print) printf("  KNN %.1f steps/sample, %.1f leaf/sample\n", double(numSteps)/numIter, double(numLeaf)/numIter);
	if(print) printf("  Average vMF threshold %.2f\n", averageVMF/m_samples.getSize());

	validateNodeBounds(ROOT, CIRCLE);		// NOTE: this really needs to be done

	if (enableCUDA)
	{
		shrinkCuda();
	} else
	{
		profilePush("Shrinking");
		Array<DensityTask> tasks2;
		tasks2.reset(NUM_DENSITY_TASKS);
		for(int i=0;i<NUM_DENSITY_TASKS;i++)
		{
			DensityTask& task = tasks2[i];
			task.init(this);
			launcher1.push(DensityTask::shrink, &task, i,1);
		}
		if(print)	launcher1.popAll("Shrinking hit splats");
		else		launcher1.popAll();
		profilePop();

		U64 numShrank   = 0;
		U64 numSelfHits = 0;
		for(int i=0;i<NUM_DENSITY_TASKS;i++)
		{
			numShrank   += tasks2[i].m_numShrank;
			numSelfHits += tasks2[i].m_numSelfHits;
		}
		if(print)					printf("  %d splats shrank\n", numShrank);
		if(print && numSelfHits)	printf("  WARNING: shrinking hit the sample itself %d times\n", (int)numSelfHits);
	}

	validateNodeBounds(ROOT, CIRCLE);

	if(print) printf("Tree grew %.1f%%\n", 100.f*(getHierarchyArea(ROOT)/origTreeArea-1));

}

//----------------------------------------------------------------------
// Filtering
//----------------------------------------------------------------------

void ReconstructIndirect::filterImage(Image& image, Image* debugImage)
{
	profilePush("Filter");
	const int h = m_sbuf->getHeight();
	MulticoreLauncher launcher;
	Array<FilterTask> ftasks;
	ftasks.reset(h);
	for(int y=0;y<h;y++)
	{
		FilterTask& ftask = ftasks[y];
		ftask.init(this,&image,debugImage);
		if(m_useDofMotionReconstruction)
		{
			launcher.push(FilterTask::filterDofMotion, &ftask, y,1);
		}
		else
		{
			if(m_rayDumpFileName.getLength())	launcher.push(FilterTask::filterPBRT, &ftask, y,1);
			else								launcher.push(FilterTask::filter, &ftask, y,1);
		}
	}
	launcher.popAll("Filtering");

	// Collect and print stats
	FilterTask::Stats stats;
	for(int y=0;y<h;y++)
		stats += ftasks[y].m_stats;

	printf("%d queries\n", (int)stats.numTraversalSteps[1]);
	printf("%.2f steps/query\n", stats.numTraversalSteps[0]/stats.numTraversalSteps[1]);
	printf("%.2f samples/query\n", stats.numSamplesTested[0]/stats.numSamplesTested[1]);
	printf("%.2f samples in R/query\n", stats.numSamplesAccepted[0]/stats.numSamplesAccepted[1]);
	printf("%.2f samples in first surface/query\n", stats.numSamplesFirstSurface[0]/stats.numSamplesFirstSurface[1]);
	printf("%.2f surfaces/query\n", stats.numSurfaces[0]/stats.numSurfaces[1]);
	printf("%d queries without support (%.4f%%)\n", int(stats.numMissingSupport[0]), 100*stats.numMissingSupport[0]/stats.numMissingSupport[1]);
	for(int i=0;i<=NUM_SAMPLE_COUNTERS;i++)
		printf("%.2f%% of queries had %d samples in first surface\n", 100*stats.numSamplesFirstSurfaceTable[i][0]/stats.numSamplesFirstSurfaceTable[i][1],i);

	printf("\n");
	printf("Average vMF support for queries %.2f\n", stats.vMFSupport[0]/stats.vMFSupport[1]);
	

	profilePop();
}

//-----------------------------------------------------------------------
// Build a hierarchy using Kontkanen et al. [2011]
//-----------------------------------------------------------------------

ReconstructIndirect::Node* ReconstructIndirect::buildRecursive(const Array<SortEntry>& codes, int& sampleIdx, const int N, const U64 octreeCode,const int octreeBitPos)
{
	if(sampleIdx >= codes.getSize())									// nothing left
		return NULL;

	const U64 mask = U64(-1) << octreeBitPos;							// relevant morton code so far
	const U64 octreeMask = octreeCode & mask;							// relevant part of octree

	bool shouldRefine = (octreeBitPos >= 3) &&							// not at max morton depth
						(codes.getSize() > sampleIdx+N) &&				// more than N samples left
						(octreeMask == (codes[sampleIdx+N].code&mask));	// N+1:th sample is inside this node

	if(!shouldRefine)
	{
		if((codes[sampleIdx].code&mask) != octreeMask)					// nothing in this octree branch
			return NULL;

		// construct a leaf node
		bool staticPoints = false;	// STATS
		bool movingPoints = false;	// STATS

		Node* node = new Node();
		Node& leaf = *node;
		leaf.s0 = sampleIdx;
		while(sampleIdx<codes.getSize() && (codes[sampleIdx].code&mask) == octreeMask)
		{
			fetchSample( m_samples[sampleIdx], codes[sampleIdx].idx );	// from sample buffer -> local struct (SLOW)
			const Sample& s = m_samples[sampleIdx++];
			const Vec3f pt0 = s.getHitPoint(0.f);
			const Vec3f pt1 = s.getHitPoint(1.f);
			leaf.bbmin   = min(leaf.bbmin,  pt0);
			leaf.bbmax   = max(leaf.bbmax,  pt0);
			leaf.bbminT1 = min(leaf.bbminT1,pt1);
			leaf.bbmaxT1 = max(leaf.bbmaxT1,pt1);
			if(pt0 == pt1)	staticPoints=true;
			else			movingPoints=true;
		}
		leaf.s1 = sampleIdx;
		leaf.ns = (leaf.s1-leaf.s0);
		leaf.bbmin   -= 1e-3f;	// needed to avoid precision issues in fast traversal code
		leaf.bbmax   += 1e-3f;
		leaf.bbminT1 += 1e-3f;
		leaf.bbmaxT1 += 1e-3f;

		if(leaf.ns>N)
		{
			printf("WARNING: %d samples in one leaf\n", leaf.ns);
			printf("octreeBitPos = %d\n", octreeBitPos);
			printf("octreeCode   = %08x%08x\n", U32(octreeCode>>32), U32(octreeCode));
		}

		// STATS
		m_totalNumMixedLeaf += (staticPoints && movingPoints) ? 1 : 0;
		m_totalNumLeafNodes++;
		m_totalNumSamples += leaf.ns;

		return node;
	}
	else
	{
		Array<Node*> nodes;
		for(int i=0;i<8;i++)
		{
			int childOctreeBitPos = octreeBitPos-3;
			U64 childOctreeCode   = octreeCode | (U64(i)<<childOctreeBitPos);
			Node* node = buildRecursive(codes, sampleIdx, N, childOctreeCode,childOctreeBitPos);
			if(node)
				nodes.add (node);
		}

		// create a balanced BVH out of this octree level (straghtforward application of Kontkanen's octree builder to BVHs).
		while(nodes.getSize() > 1)
		{
			Array<Node*> nodes2;

			int k=0;
			for(;k+1<nodes.getSize();k+=2)
			{
				// Perform merge and emit node
				Node* n0 = nodes[k];
				Node* n1 = nodes[k+1];
				Node* n = new Node(*n0,*n1);
				n->child0 = m_hierarchy.getSize();	m_hierarchy.add(*n0);
				n->child1 = m_hierarchy.getSize();	m_hierarchy.add(*n1);
				nodes2.add(n);
			}
			if(k<nodes.getSize())
				nodes2.add(nodes.getLast());

			nodes = nodes2;
		}

		return nodes.getSize() ? nodes[0] : NULL;
	}
}

float ReconstructIndirect::getHierarchyArea(int nodeIdx, bool leafOnly) const
{
	float time = 0.f;	// @ t=0
	const Node& node = m_hierarchy[nodeIdx];
	if(node.isLeaf())	return node.getSurfaceArea(time);
	else				return (leafOnly ? 0 : node.getSurfaceArea(time)) + getHierarchyArea(node.child0) + getHierarchyArea(node.child1);
}

void ReconstructIndirect::validateNodeBounds(int nodeIdx, BloatMode mode)
{
	Node& node = m_hierarchy[nodeIdx];

	if(node.isLeaf())
	{
		node.bbmin   = Vec3f( FW_F32_MAX);
		node.bbmax   = Vec3f(-FW_F32_MAX);
		node.bbminT1 = Vec3f( FW_F32_MAX);
		node.bbmaxT1 = Vec3f(-FW_F32_MAX);

		for(int i=node.s0;i<node.s1;i++)
		{
			const Sample& s = m_samples[i];
			const Vec3f& n = s.sec_normal;
			const float& R = s.radius;
			Vec3f ext(1e-3f);									// needed to avoid precision issues in fast traversal code

			switch(mode)
			{
			case CIRCLE:
				{
					const float cosx = dot(Vec3f(1,0,0), n);
					const float cosy = dot(Vec3f(0,1,0), n);
					const float cosz = dot(Vec3f(0,0,1), n);
					const float sinx = sqrt(1-cosx*cosx);		// sin^2 + cos^2 = 1
					const float siny = sqrt(1-cosy*cosy);
					const float sinz = sqrt(1-cosz*cosz);
					ext += R * Vec3f(sinx,siny,sinz);
					break;
				}
			case SPHERE:
				{
					ext += R;
					break;
				}
			case POINT:
				{
					break;
				}
			default:
				fail("validateNodeBounds");
			} // switch

			const Vec3f  pt0 = s.getHitPoint(0.f);				// @ t=0
			const Vec3f  pt1 = s.getHitPoint(1.f);				// @ t=1
			node.bbmin   = min(node.bbmin, pt0-ext);
			node.bbmax   = max(node.bbmax, pt0+ext);
			node.bbminT1 = min(node.bbminT1, pt1-ext);
			node.bbmaxT1 = max(node.bbmaxT1, pt1+ext);
		}
	}
	else
	{
		const Node& c0 = m_hierarchy[node.child0];
		const Node& c1 = m_hierarchy[node.child1];
		validateNodeBounds(node.child0, mode);
		validateNodeBounds(node.child1, mode);
		node.bbmin   = min(c0.bbmin,c1.bbmin);
		node.bbmax   = max(c0.bbmax,c1.bbmax);
		node.bbminT1 = min(c0.bbminT1,c1.bbminT1);
		node.bbmaxT1 = max(c0.bbmaxT1,c1.bbmaxT1);
	}
}

//-----------------------------------------------------------------------
// Verify that all of the hitpoints can be found by traversing the tree
//-----------------------------------------------------------------------

void ReconstructIndirect::DensityTask::compute(int taskIdx)
{
	const int EMERGENCY_BREAK_LIM = 1000;		// Stop searching if not found the requested kind of samples within this amount of work
	const float VMF_THRESHOLD_MAX = 0.5f;		// We're really happy if we get this, but if not, we'll lower it until things work out
	const float VMF_THRESHOLD_SCALE = 0.75f;

	const bool useBandwithInformation = this->m_scope->m_useBandwidthInformation;

	if(taskIdx<0 || taskIdx>=NUM_DENSITY_TASKS)
		fail("ReconstructIndirect::DensityTask::compute");

	struct PQEntry
	{
		float	key;		// distance
		int		index;		// in m_samples
		bool	operator<(const PQEntry& s)	{ return key>s.key; }
	};

	struct StackEntry
	{
		StackEntry()							{ }
		StackEntry(int i,float d)				{ index=i; dist=d; }
		bool	operator<(const StackEntry& s)	{ return dist<s.dist; }
		int		index;		// m_hierarchy
		float	dist;		// distance to node
	};

		  Array<Sample>& samples = m_scope->m_samples;
	const Array<Node>& hierarchy = m_scope->m_hierarchy;

	m_numSteps = 0;
	m_numLeaf  = 0;
	m_numIter  = 0;
	m_averageVMF = 0;

	Array<StackEntry> stack;
	BinaryHeap<StackEntry> stack2;		// prioritized traversal stack
	BinaryHeap<PQEntry>	pq;				// for K nearest points
	Array<float> distances;

	const float t = 0.5f;				// compute @ t=0.5
	const int TASK_SIZE = (samples.getSize()+NUM_DENSITY_TASKS-1) / NUM_DENSITY_TASKS;	// #samples in each task, rounded up
	const int lo = taskIdx*TASK_SIZE;													// inc
	const int hi = min(lo+TASK_SIZE, samples.getSize());								// exc
	for(int i=lo;i<hi;i++)
	{
		m_numIter++;
		const Sample& s = samples[i];
		const Vec3f   o = s.getHitPoint(t);
		const Vec3f&  n = s.sec_normal;

		const Mat3f cameraToTangentplane = orthogonalBasis(n).transposed();				// inverse (symmetric matrix)

		// Step 1: Collect K nearest samples from the tree

		for(float vMFThreshold=VMF_THRESHOLD_MAX; vMFThreshold>=0.01f; vMFThreshold*=VMF_THRESHOLD_SCALE)
		{
			float thresholdDist = FW_F32_MAX;
			pq.clear();
			stack2.clear();
			stack2.add( StackEntry(ROOT,0.f) );

			// Try with a specific vMF threshold
			int numSamplesTested = 0;
			while(stack2.numItems() && numSamplesTested<EMERGENCY_BREAK_LIM)
			{
				m_numSteps++;
				StackEntry se = stack2.removeMin();
				if(se.dist >= thresholdDist)
					break;

				const Node& node = hierarchy[se.index];
				if(node.isLeaf())
				{
					m_numLeaf++;
					for(int j=node.s0;j<node.s1;j++)
					{
						const Vec3f  p   = samples[j].getHitPoint(t);
						const Vec3f td   = cameraToTangentplane * (o-p);					// p->o in tangent plane's coordinate system (z aligned to normal)
						float dist = (td*Vec3f(1,1,1+ANISOTROPIC_SCALE)).length();	// anisotropic scale

						if(useBandwithInformation)
						{
							numSamplesTested++;
							double anglecos = dot( (s.sec_origin-s.sec_hitpoint).normalized(), (samples[j].sec_origin-samples[j].sec_hitpoint).normalized() );
							double bw = FilterTask::vMFfromBandwidth( m_scope->m_sbuf->getSampleW( s.origIndex.x, s.origIndex.y, s.origIndex.z ) );
							double vMF = exp( bw * anglecos - bw );	// non-normalized vMF, in [0,1]
							if ( vMF < vMFThreshold )
								continue;
						}

						// update per-sample priority queue
						PQEntry e;
						e.index = j;
						e.key   = dist;
						if(pq.numItems()==K1)
						{
							const PQEntry& me = pq.getMin();
							if(e.key < me.key)
							{
								pq.removeMin();	// replace previous _largest_ (of K smallest)
								pq.add( e );
							}
						}
						else
						{
							pq.add( e );		// add
						}
					}

					thresholdDist = (pq.numItems()==K1) ? pq.getMin().key : FW_F32_MAX;	// anything larger than this is of no interest
				}
				else
				{
					StackEntry se0( node.child0, hierarchy[node.child0].getDistance(o,t) );
					StackEntry se1( node.child1, hierarchy[node.child1].getDistance(o,t) );
					stack2.add( se0 );
					stack2.add( se1 );
				}
			}

			// did we succeed?
			if(numSamplesTested<EMERGENCY_BREAK_LIM)
			{
				m_averageVMF += vMFThreshold;
				break;
			}
		}

		// Step 2: Determine a sample's radius on its tangent plane from K2 nearest samples on the plane

		// Project the nearest points to the plane defined by sample i
		//
		// (p+t*n,1) | plane = 0
		// t*n | plane.xyz = -(p,1) | plane
		// t = -(p,1) | plane / (n | plane.xyz)

		if(K2 < K1)
		{
			distances.clear();
			const Vec4f plane = Vec4f(n, -dot(n,o));	// ith sample lies on this plane
			const float oon   = rcp( dot(n,plane.getXYZ()) );
			while(pq.numItems())
			{
				const Vec3f p   = samples[ pq.removeMin().index ].getHitPoint(t);	// in 3D
				const float t   = -dot(Vec4f(p,1),plane) * oon;
				const Vec3f pip = p+t*n;											// projected to the plane along o's normal
				distances.add( (o-pip).length() );
			}

			FW_SORT_ARRAY( distances, float, a < b );
			samples[i].radius = distances[K2];
		}
		else
		{
			const Vec3f p   = samples[ pq.removeMin().index ].getHitPoint(t);		// in 3D
			samples[i].radius = (p-o).length();										// isotropic
		}
	}
}

void ReconstructIndirect::DensityTask::shrink(int taskIdx)
{
	if(taskIdx<0 || taskIdx>=NUM_DENSITY_TASKS)
		fail("ReconstructIndirect::DensityTask::shrink");

	      Array<Sample>& samples  = m_scope->m_samples;
	const Array<Node>&	hierarchy = m_scope->m_hierarchy;

	Array<int> stack;

	m_numShrank = 0;
	m_numSelfHits = 0;

	const int TASK_SIZE = (samples.getSize()+NUM_DENSITY_TASKS-1) / NUM_DENSITY_TASKS;	// #samples in each task, rounded up
	const int lo = taskIdx*TASK_SIZE;													// inc
	const int hi = min(lo+TASK_SIZE, samples.getSize());								// exc
	for(int i=lo;i<hi;i++)
	{
		// we know this ray didn't hit anything
		const Sample& sa = samples[i];
		const float time = sa.t;
		const Vec3f orig = sa.sec_origin;
		const Vec3f hitp = sa.getHitPoint(time);

		const float rayLen = (hitp-orig).length();
		const Vec3f dir    = (hitp-orig) / rayLen;	// unit length

		// prepare the ray for fast traversal
        const float ooeps = 1e-20f;
		Vec3f idir;
        idir.x = 1.0f / (fabs(dir.x)>ooeps ? dir.x : (dir.x<0 ? -ooeps : ooeps));
        idir.y = 1.0f / (fabs(dir.y)>ooeps ? dir.y : (dir.y<0 ? -ooeps : ooeps));
        idir.z = 1.0f / (fabs(dir.z)>ooeps ? dir.z : (dir.z<0 ? -ooeps : ooeps));
		const Vec3f ood = orig * idir;

		// adaptive epsilon similar to from PBRT (takes max because for defocus the rays start from the origin of camera space)
		const float eps = 1e-3f * max(orig.length(),hitp.length());

		stack.clear();
		stack.add(ROOT);
		while(stack.getSize())
		{
			const int nodeIndex = stack.removeLast();
			const Node& node = hierarchy[nodeIndex];

			if(node.intersect(idir,ood,time))
			{
				if(node.isLeaf())
				{
					for(int j=node.s0;j<node.s1;j++)
					{
						const Sample& s = samples[j];
						const Vec3f& p = s.getHitPoint(time);

						float t;
						const Vec3f tp = intersectRayPlane(t, orig,dir, s.getTangentPlane(time));
						const float tpDist = (tp-p).length();			// distance on the tangent plane
						if(tpDist >= s.radius)							// (extended) ray doesn't hit the splat
							continue;

						if(t<=eps || t>=rayLen-eps)						// avoid accidental hits very close to origin and hitpoint
							continue;

						if(i==j)
						{
							m_numSelfHits++;
							continue;
						}

						samples[j].radius = tpDist;						// NOTE: theoretically this doesn't work with multicore, probably OK in practice
						//samples[j].color = Vec3f(0,0,1);				// DEBUG visualization: make the shrank splats blue
						m_numShrank++;
					}
				}
				else
				{
					stack.add( node.child0 );
					stack.add( node.child1 );
				}
			}
		}
	}
}

//-----------------------------------------------------------------------
// Filter scanline (The main reconstruction loop, does indirect/AO)
//-----------------------------------------------------------------------

// DEBUG visualizations: hemisphere for (x,y). If defined exports 512x512 hemisphere.png to the current folder.
//#define EXPORT_HEMISPHERE (x==246 && y==430)

void ReconstructIndirect::FilterTask::filter(int y)
{
	const UVTSampleBuffer& sbuf = *m_scope->m_sbuf;

	const int w = sbuf.getWidth ();
	const int h = sbuf.getHeight();
	const int n = sbuf.getNumSamples()/SUBSAMPLE_SBUF;
	FW_UNREF(h);
	FW_UNREF(n);

	const int xmin = this->m_scope->m_scissor[0];
	const int xmax = this->m_scope->m_scissor[2];
	const int ymin = this->m_scope->m_scissor[1];
	const int ymax = this->m_scope->m_scissor[3];

	Random random(y);

	for(int x=0;x<w;x++)
	{
		Vec2f duv(random.getF32(),random.getF32());

	#ifdef EXPORT_HEMISPHERE
		if(!EXPORT_HEMISPHERE)
			continue;
		const Vec2i hemisphereSize(512,512);
		Image hemisphere(hemisphereSize, ImageFormat::RGBA_Vec4f);
		hemisphere.clear(0);
	#endif

		if(!(x>=xmin && x<=xmax && y>=ymin && y<=ymax))	// outside the scissor
			continue;

		if(x==xmin || x==xmax || y==ymin || y==ymax)	// make scissor border white
		{
			if(x>=0 && y>=0 && x<w && y<h)				// inside drawable area?
				m_image->setVec4f(Vec2i(x,y),Vec4f(1));
			continue;
		}

		//---------------------------------------------------------------

		clearNumUniqueInputSamplesUsed();

		Vec4f pixelColor(0);
		m_pixelIndex = Vec2i(x,y);
		int numHasSupport = 0;
		int	numNoSupport = 0;
		int i=0;

		const bool ambientOcclusion = (this->m_scope->m_aoLength>0);

	#ifdef EXPORT_HEMISPHERE
		const int N = hemisphereSize.x * hemisphereSize.y;
	#else
		// baseline scenario
		const int N = m_scope->m_numReconstructionRays / n;		// # samples to take
		for( ;i<n;i++)		
		if(isSecondaryOriginValid(sbuf,x,y,i))
	#endif
		{
			// data at the origin of the secondary ray
			const float eps = 1e-3f;																// TODO: theoretically this could be 0 (and apparently is in PBRT)
			const Vec3f normal = sbuf.getSampleExtra<Vec3f>(CID_PRI_NORMAL,  x,y,i);				// orientation of hemisphere
			const Vec3f origin = sbuf.getSampleExtra<Vec3f>(CID_SEC_ORIGIN,  x,y,i) + eps*normal;	// shoot secondary from here
			      Vec3f albedo = sbuf.getSampleExtra<Vec3f>(CID_ALBEDO,      x,y,i);				// albedo (needed once incident light has been computed)
			const Mat3f unitHemisphereToCamera = orthogonalBasis(normal);							// hemisphere -> camera coordinates

			if(ambientOcclusion)
				albedo = 1.f;

			// cast N rays from each origin
			for(int j=0;j<N;j++)
			{
				m_stats.newOutput();

				// generate ray

#ifdef EXPORT_HEMISPHERE
				const int hsx = j%hemisphereSize.x;		// [0,w)
				const int hsy = j/hemisphereSize.x;		// [0,h)
				const Vec2f disk = Vec2f(float(hsx)/hemisphereSize.x, float(hsy)/hemisphereSize.y)*2-1;	// [-1,1]
				if(disk.length()>=1.f)
					continue;
				const Vec3f dunit = diskToCosineHemisphere(disk);
#else
				// baseline scenario
				Vec2f square( sobol(2,j+i*N),sobol(3,j+i*N) );	// [0,1]
				CranleyPatterson(square,duv);
				const Vec3f dunit = (ambientOcclusion) ? squareToCosineHemisphere(square) : squareToCosineHemisphere(square);
#endif

				// direction vector in camera space
				Vec3f direction = (unitHemisphereToCamera * dunit).normalized();

				// sample
				const Vec4f incidentRadiance = sampleRadiance(origin,direction);

				// accumulate
				if(incidentRadiance.w == 0)
				{
					numNoSupport++;
				}
				else
				{
					numHasSupport++;
					pixelColor += Vec4f(albedo,1) * incidentRadiance;	// outgoing radiance
				}

#ifdef EXPORT_HEMISPHERE
				hemisphere.setVec4f(Vec2i(hsx,hsy), (incidentRadiance.w) ? incidentRadiance : Vec4f(0,1,0,1));
#endif
			} // j
		} // samples
		
		m_image->setVec4f(Vec2i(x,y),pixelColor*rcp(pixelColor.w));

		if(m_debugImage)
		{
			Vec4f debugColor(0,0,0,1);
			debugColor.x = float(numHasSupport)/(numHasSupport+numNoSupport);			// white = OK, black = total lack of support
			debugColor.y = debugColor.x;
			debugColor.z = debugColor.x;
			m_debugImage->setVec4f(Vec2i(x,y),debugColor);
		}

		m_stats.numMissingSupport[0] += numNoSupport;

#ifdef EXPORT_HEMISPHERE
		exportImage("hemisphere.png", &hemisphere);
		failIfError();
#endif
	} // pixels
}

//-----------------------------------------------------------------------
// Filter scanline (Glossy, PBRT's ray dump)
//-----------------------------------------------------------------------

void ReconstructIndirect::FilterTask::filterPBRT(int y)
{
	const UVTSampleBuffer& sbuf = *m_scope->m_sbuf;
	const Array64<PBRTReconstructionRay>& PBRTReconstructionRays = m_scope->m_PBRTReconstructionRays;

	const int w = sbuf.getWidth ();
	const int h = sbuf.getHeight();
	const int n = sbuf.getNumSamples()/SUBSAMPLE_SBUF;
	FW_UNREF(h);
	FW_UNREF(n);

	const int xmin = this->m_scope->m_scissor[0];
	const int xmax = this->m_scope->m_scissor[2];
	const int ymin = this->m_scope->m_scissor[1];
	const int ymax = this->m_scope->m_scissor[3];

	// DEBUG DEBUG
	const int MAX_N_PER_PIXEL = 1024;

	// clear scanline
	for(int x=0;x<w;x++)
		m_image->setVec4f(Vec2i(x,y),Vec4f(0,1,0,1));

	// have rays to trace?
	int rayIndex = m_scope->m_PBRTReconstructionRaysScanlineStart[y];
	if(rayIndex==-1)
		return;

	Vec2i currentPixel(-1);
	Vec4f pixelColor(0);
	Vec2f vMFSupport(0);
	float vMFMinAngle =  FW_F32_MAX;
	float vMFMaxAngle = -FW_F32_MAX;
	Vec2f vMFAvgAngle(0);
	int numHasSupport = 0;
	int	numNoSupport = 0;

	for( ;rayIndex<PBRTReconstructionRays.getSize();rayIndex++)
	{
		const PBRTReconstructionRay& ray = PBRTReconstructionRays[ rayIndex ];
		if((int)floor(ray.xy[1]) != y)
			return;

		const int x = (int)floor(ray.xy[0]);
		const Vec2i pixel(x,y);

		if(!(x>=xmin && x<=xmax && y>=ymin && y<=ymax))	// outside the scissor
			continue;

		if(x==xmin || x==xmax || y==ymin || y==ymax)	// make scissor border white
		{
			if(x>=0 && y>=0 && x<w && y<h)				// inside drawable area?
				m_image->setVec4f(Vec2i(x,y),Vec4f(1));
			continue;
		}

		if(pixel != currentPixel)
		{
			currentPixel = pixel;
			pixelColor = 0.f;
			vMFSupport = 0.f;
			vMFMinAngle =  FW_F32_MAX;
			vMFMaxAngle = -FW_F32_MAX;
			vMFAvgAngle = 0.f;
			numHasSupport = 0;
			numNoSupport = 0;
			m_pixelIndex = pixel;
			clearNumUniqueInputSamplesUsed();
		}
		else
		{
			if(pixelColor.w >= MAX_N_PER_PIXEL)
				continue;
		}

		// sample radiance

		m_stats.newOutput();

		const Vec3f& origin    = ray.o;
		const Vec3f& direction = ray.d;
		const Vec3f& weight    = ray.weight;

		Vec4f incidentRadiance(0,0,0,1);			// black if weight = 0 (below horizon)

		if(weight!=Vec3f(0))
		{
			incidentRadiance = sampleRadiance(origin,direction);

			m_stats.vMFSupport[0] += m_vMFSupport;	// DEBUG DEBUG (set by sampleRadiance);
			vMFSupport += Vec2f(m_vMFSupport,1);
			vMFMinAngle = min(vMFMinAngle,m_vMFAngle);
			vMFMaxAngle = max(vMFMaxAngle,m_vMFAngle);
			vMFAvgAngle+= Vec2f(m_vMFAngle,1);
		}

		// accumulate
		if(incidentRadiance.w == 0)
		{
			numNoSupport++;
			m_stats.numMissingSupport[0]++;
			//pixelColor += Vec4f(1,0,0,1);			// DEBUG DEBUG
		}
		else
		{
			numHasSupport++;
			pixelColor += Vec4f(weight*incidentRadiance.getXYZ(),1);	// outgoing radiance
		}

		// update images
		m_image->setVec4f( pixel,pixelColor*rcp(pixelColor.w) );
		if(m_debugImage)
		{
			Vec4f debugColor(0,0,0,1);
			//debugColor.x = float(numHasSupport)/(numHasSupport+numNoSupport);	// white = OK, black = total lack of support
			//debugColor.x = vMFSupport.x/vMFSupport.y;							// white = good support, black = failure
			debugColor.x = vMFMinAngle;
			debugColor.y = vMFMaxAngle;
			debugColor.z = vMFAvgAngle.x/vMFAvgAngle.y;
			m_debugImage->setVec4f(Vec2i(x,y),debugColor);
		}
	}
}

//-----------------------------------------------------------------------
// Filter scanline (dof, motion)
//-----------------------------------------------------------------------

void ReconstructIndirect::FilterTask::filterDofMotion(int y)
{
	const UVTSampleBuffer& sbuf = *m_scope->m_sbuf;

	const int w = sbuf.getWidth ();
	const int h = sbuf.getHeight();
	const int n = sbuf.getNumSamples()/SUBSAMPLE_SBUF;
	FW_UNREF(h);
	FW_UNREF(n);

	Mat4f screenToFocusPlane;
	screenToFocusPlane.set(sbuf.getPixelToFocalPlaneMatrix());
	screenToFocusPlane.transpose();

	const float lensRadius = fabs(sbuf.getCocCoeffs()[0] * screenToFocusPlane.get(0,0) / screenToFocusPlane.get(2,3));

	const int xmin = this->m_scope->m_scissor[0];
	const int xmax = this->m_scope->m_scissor[2];
	const int ymin = this->m_scope->m_scissor[1];
	const int ymax = this->m_scope->m_scissor[3];

	Random random(y);

	for(int x=0;x<w;x++)
	{
		// random offsets for Cranley-Patterson
		const float dt  (random.getF32());

		if(!(x>=xmin && x<=xmax && y>=ymin && y<=ymax))	// outside the scissor
			continue;

		if(x==xmin || x==xmax || y==ymin || y==ymax)	// make scissor border white
		{
			if(x>=0 && y>=0 && x<w && y<h)				// inside drawable area?
				m_image->setVec4f(Vec2i(x,y),Vec4f(1));
			continue;
		}

		clearNumUniqueInputSamplesUsed();

		Vec4f pixelColor(0);
		m_pixelIndex = Vec2i(x,y);
		int numHasSupport = 0;
		int	numNoSupport = 0;

		const U32 mortoncode = (U32)morton(x,y);

		const int N = m_scope->m_numReconstructionRays;
		for(int i=0;i<N;i++)
		{
			m_stats.newOutput();

			// QMC
			const int idx = mortoncode*N+i;
			Vec2f square1( sobol(0,idx),sobol(2,idx) );	// [0,1]	uv
			Vec2f square2( sobol(3,idx),sobol(4,idx) );	// [0,1]	xy
			float time = hammersley(i,N);
			CranleyPatterson( time,dt );				// [0,1]

			const Vec2f disk = lensRadius * squareToDisk(square1);	// [-R,R]

			// generate ray
			Vec3f onFocusPlane = (screenToFocusPlane * Vec4f(x+square2.x,y+square2.y,0,1)).toCartesian();
			const Vec3f origin( disk, 0 );
			const Vec3f direction = (onFocusPlane-origin).normalized();

			// sample
			const Vec4f incidentRadiance = sampleRadiance(origin,direction,time);

			// accumulate
			if(incidentRadiance.w == 0)
			{
				numNoSupport++;
			}
			else
			{
				numHasSupport++;
				pixelColor += incidentRadiance;	// outgoing radiance
			}
		} // samples
		
		m_image->setVec4f(Vec2i(x,y),pixelColor*rcp(pixelColor.w));

		if(m_debugImage)
		{
			Vec4f debugColor(0,0,0,1);
			debugColor.x = float(numHasSupport)/(numHasSupport+numNoSupport);			// white = OK, black = total lack of support
			debugColor.y = debugColor.x;
			debugColor.z = debugColor.x;
			m_debugImage->setVec4f(Vec2i(x,y),debugColor);
		}

		m_stats.numMissingSupport[0] += numNoSupport;
	} // pixels
}

//-----------------------------------------------------------------------
// Compute radiance for a given ray
//-----------------------------------------------------------------------

Vec4f ReconstructIndirect::FilterTask::sampleRadiance(const Vec3f& o,const Vec3f& d,float time)
{
	// setup local parameterization
	LocalParameterization lp(o,d,time);

	// gather samples from the tree
	collectSamples(lp);

	// filter the samples
	Array<ReconSample>& rs = m_reconSamples;
	Vec4f sampleColor(0);
	Vec2i samplesInSurface(0,0);
	int	prevProcessedSample = 0;	// used for merging small surfaces to the next
	while(samplesInSurface[1] != rs.getSize())
	{
		samplesInSurface = getNextSurface( samplesInSurface,lp );

		const bool firstSurface = (samplesInSurface[0] == 0);
		const bool lastSurface  = (samplesInSurface[1] == rs.getSize());

		m_stats.numSamplesFirstSurfaceTable[min(samplesInSurface[1]-prevProcessedSample,(int)NUM_SAMPLE_COUNTERS)][0]++;

		if(firstSurface)
			m_stats.numSamplesFirstSurface[0] += (samplesInSurface[1]-samplesInSurface[0]);

		if(!lastSurface)
		{
			// merge small surfaces to the next 
			const int SMALL_SURFACE_LIMIT = 4;

			if(samplesInSurface[1]-prevProcessedSample<SMALL_SURFACE_LIMIT)
			{
				if(!isOriginInsideConvexHull(prevProcessedSample,samplesInSurface[1], lp))
					continue;
			}
		}

		// for glossy surfaces
		const bool selectNearestSample = this->m_scope->m_selectNearestSample;
		if(selectNearestSample)
		{
			float maxWeight = -FW_F32_MAX;
			int maxIndex = -1;
			for(int i=prevProcessedSample;i<samplesInSurface[1];i++)
			{
				const ReconSample& r = rs[i];
				if(r.weight > maxWeight)
				{
					maxWeight = r.weight;
					maxIndex = i;
				}
			}

			if(maxIndex != -1)
			{
				m_vMFSupport = maxWeight;
				const Sample& s = m_scope->m_samples[ rs[maxIndex].index ];
				const float cosangle = dot(lp.dir,(s.getHitPoint(lp.time)-s.sec_origin).normalized());	// 0,1
				m_vMFAngle = acos( clamp(cosangle,0.f,1.f) ) / (FW_PI/2);	// angle between the vectors [0,1]
			}

			for(int i=prevProcessedSample;i<samplesInSurface[1];i++)
			{
				rs[i].weight = (i == maxIndex) ? 1.f : 0.f;
				//rs[i].color = m_scope->m_samples[ rs[i].index ].size;
			}
		}

		const float aoLength = (this->m_scope->m_aoLength);
		for(int i=prevProcessedSample;i<samplesInSurface[1];i++)
		{
			const ReconSample& r = rs[i];
			Vec3f color = r.color;

			if(aoLength>0)
				color = (r.zdist<=aoLength) ? 0.f : 1.f;

			sampleColor += r.weight * Vec4f(color,1);

			if( r.weight > 0.0f && !m_supportSet.contains( r.index ))
				m_supportSet.add( r.index );
		}

		prevProcessedSample = samplesInSurface[1];

		// found something?
		if(sampleColor.w)
			break;
	}

	return sampleColor * rcp(sampleColor.w);
}

void ReconstructIndirect::FilterTask::collectSamples(const LocalParameterization& lp)
{
	const Array<Sample>& samples  = m_scope->m_samples;
	const Array<Node>&	hierarchy = m_scope->m_hierarchy;

	const Vec3f& idir   = lp.idir;
	const Vec3f& ood    = lp.ood;
	const Vec3f& orig   = lp.orig;
	const Vec3f& dir    = lp.dir;
	const float  time   = lp.time;
	FW_UNREF(dir);

	Array<int>& stack = m_stack;
	Array<ReconSample>& rs = m_reconSamples;

	const bool useBandwidthInformation = this->m_scope->m_useBandwidthInformation;

	//----------------------------------------------------------------------
	// collect intersected splats from the tree
	//----------------------------------------------------------------------

	rs.clear();
	stack.clear();
	stack.add(ROOT);

	while(stack.getSize())
	{
		m_stats.numTraversalSteps[0]++;
		const int nodeIndex = stack.removeLast();
		const Node& node = hierarchy[nodeIndex];
		const float eps = 1e-3f * orig.length();		// PBRT epsilon: 1e-3f * distance of previous ray. We're in camera space, meaning the primary ray is (0,0,0)->lp.orig...

		if(node.isLeaf())
		{
			m_stats.numSamplesTested[0] += node.ns;
			for(int sidx=node.s0;sidx<node.s1;sidx++)
			{
				const Sample& s = samples[sidx];		// a sample whose SECONDARY HITPOINT is near our query ray
				const Vec3f& p = s.getHitPoint(time);

				// intersection point on the sample's tangent plane
				float t;
				const Vec3f tp = intersectRayPlane(t, orig,dir, s.getTangentPlane(time));	// intersection point on sample's tangent plane
				const float f  = (tp-p).length() / s.radius;

				// intersects the splat and t>0
				if(f<1.f && t>eps)
				{
					const bool backface = dot(s.sec_normal, p-orig) >= 0;

	#ifdef ENABLE_BACKFACE_CULLING
					if(backface)
						continue;
	#endif
					const float distFromST = dot(lp.stplane,Vec4f(p,1));			// Splat center's distance from ST plane

					// back-face in the nearfield -> cull (basically because small concavities suffer from undersampling)
					if(backface && distFromST<s.radius)
						continue;

					m_stats.numSamplesAccepted[0]++;
					ReconSample& r = rs.add();
					r.backface = backface;
					r.zdist = distFromST;
					r.color = (backface) ? Vec3f(0,0,0) : s.color;					// Back-facing splats are black
					if(useBandwidthInformation)
						r.weight = interpolationWeight( orig, dir, tp, s );
					else
						r.weight= 1-f;
					r.index = sidx;
				}
			}
		}
		else
		{
			// intersect child nodes (since we collect all samples, sorting wouldn't help)
			const int nodeIdx0 = node.child0;
			const int nodeIdx1 = node.child1;
			if( hierarchy[nodeIdx0].intersect(idir,ood,time) )	stack.add( nodeIdx0 );
			if( hierarchy[nodeIdx1].intersect(idir,ood,time) )	stack.add( nodeIdx1 );
		}
	}

	//----------------------------------------------------------------------
	// Sort samples to increasing depth order
	//----------------------------------------------------------------------

	FW_SORT_ARRAY(rs,ReconSample, a.zdist < b.zdist);
}

//-------------------------------------------------------------------
// Extract next surface
//-------------------------------------------------------------------

Vec2i ReconstructIndirect::FilterTask::getNextSurface(Vec2i samplesInPrevSurface,const LocalParameterization& lp)
{
	FW_UNREF(lp);

	const bool SEPARATE_SURFACES_SIMPLE   = false;	// select one or neither
	const bool SEPARATE_SURFACES_SPECTRUM = true;	// 

		  Array<ReconSample>& rs = m_reconSamples;
	const Array<Sample>& samples = m_scope->m_samples;

	if( samplesInPrevSurface[0] >= rs.getSize() )						// already processed all (shouldn't have called this function)
		return Vec2i(rs.getSize());

	Vec2i samplesInSurface(samplesInPrevSurface[1],rs.getSize());		// default: all the rest

	//-------------------------------------------------------------------
	// Simple depth threshold-based query
	//-------------------------------------------------------------------

	if(SEPARATE_SURFACES_SIMPLE)
	{
		const float dR = 5.f;											// differential radius on x-plane (currently: approx. average minimum distance of secondary ray origins)
		float prevCoc = 0;
		for(int i=samplesInSurface[0];i<samplesInSurface[1];i++)
		{
			const float coc = dR*(rs[i].zdist-UVPLANE_DISTANCE);
			const float mn = max(SUBSAMPLE_SBUF*5*dR,min(coc,prevCoc));	// max(JOUNI,...) forces everything very close to be part of the same surface (this is a HACK to get Cornell working)
			const float mx = max(SUBSAMPLE_SBUF*5*dR,max(coc,prevCoc));
			if(i>samplesInSurface[0] && mx/mn>1.1f)
				samplesInSurface[1] = i;
			prevCoc = coc;
		}
	}
	
	//-------------------------------------------------------------------
	// Lightfield crossings
	//-------------------------------------------------------------------
	
	if(SEPARATE_SURFACES_SPECTRUM)
	{
		// SIGGRAPH11-style SameSurface()
		// O(n^2): loop between all pairs of samples

		const float time = lp.time;
		bool bConflict = false;
		for(int i=samplesInSurface[0]; i<samplesInSurface[1] && !bConflict; i++)
		{
			// any crossings with samples already in the current surface?
			for(int j=samplesInSurface[0]; j<i && !bConflict; j++)
			{
				// a pair of samples
				const ReconSample& r1 = rs[i];
				const ReconSample& r2 = rs[j];
				const Sample& sa1 = samples[ r1.index ];
				const Sample& sa2 = samples[ r2.index ];

				// two-sided primitives (as in PBRT): flip the normal
				const Vec3f normal1 = r1.backface ? -sa1.sec_normal : sa1.sec_normal;
				const Vec3f normal2 = r2.backface ? -sa2.sec_normal : sa2.sec_normal;

				// OPTION 1: consistently facing toward or away from each other?
				// same surface if
				// - angle1>=0 && angle2>=0
				// - angle1<=0 && angle2<=0
				// - z difference is smaller than the min/average size of splats (not sure how exactly to do this)

				const float cosAngle1 = dot(normal1, (sa2.getHitPoint(time)-sa1.getHitPoint(time)).normalized());	// cos(n1,sep)
				const float cosAngle2 = dot(normal2, (sa1.getHitPoint(time)-sa2.getHitPoint(time)).normalized());
				const float eps = sin(2*FW_PI/180);
				if( (cosAngle1+eps)>=0 && (cosAngle2+eps)>=0 )	continue;
				if( (cosAngle1-eps)<=0 && (cosAngle2-eps)<=0 )	continue;
//				if( fabs(rs[i].zdist-rs[j].zdist) < (sa1.size+sa2.size)/2 )	continue;
				if( fabs(dot(sa1.getTangentPlane(time),Vec4f(sa2.getHitPoint(time),1))) < min(sa1.radius,sa2.radius) )	continue;
				bConflict = true;

/*				// OPTION 2: alternative interpretation, which is explained in the paper
				const Vec3f p1 = sa1.getHitPoint(time);
				const Vec3f p2 = sa2.getHitPoint(time);
				const Vec4f plane1(normal1, -dot(normal1,p1));
				const Vec4f plane2(normal2, -dot(normal2,p2));
				const Vec3f sep  = p2-p1;	// 1->2

				const Vec3f pip = intersectRayPlane(p1,sep, lp.stplane);
				const float eps = 1e-3f;
				if((dot(plane1,Vec4f(pip,1))> eps && dot(plane2,Vec4f(pip,1))> eps) || 	// pip in ++
				   (dot(plane1,Vec4f(pip,1))<-eps && dot(plane2,Vec4f(pip,1))<-eps) )	// pip in --
					bConflict = true;
/**/
			}

			if(bConflict)
				samplesInSurface[1] = i;
		}
	}

	return samplesInSurface;
}


} // namespace
