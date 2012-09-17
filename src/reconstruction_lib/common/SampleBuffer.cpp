#pragma warning(disable:4127)
#pragma warning(disable:4996)

#include "SampleBuffer.hpp"
#include "Util.hpp"
#include "base/Sort.hpp"
#include "gui/Image.hpp"
#include "3d/ConvexPolyhedron.hpp"
#include <cstdio>

namespace FW
{

SampleBuffer::SampleBuffer(int w,int h, int numSamplesPerPixel)
{
	m_width  = w;
	m_height = h;
	m_numSamplesPerPixel = numSamplesPerPixel;
	m_irregular = false;

	m_color .reset(m_width*m_height*m_numSamplesPerPixel);
	m_depth .reset(m_width*m_height*m_numSamplesPerPixel);
	m_w     .reset(m_width*m_height*m_numSamplesPerPixel);
    m_weight.reset(m_width*m_height*m_numSamplesPerPixel);

	// Generate XY samples

	Random random(242);
	m_xy.reset(m_width*m_height*m_numSamplesPerPixel);
	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	{
		const bool XY_HALTON     = false;	// select one (very minor effect on image quality)
		const bool XY_HAMMERSLEY = false;
		const bool XY_SOBOL	     = true;
		const bool XY_LP		 = false;

		Vec2f offset(random.getF32(),random.getF32());	// [0,1)
		if(m_numSamplesPerPixel<=4)
			offset = 0;		// TODO

		for(int i=0;i<m_numSamplesPerPixel;i++)
		{
			int j=i+1;	// 0 produces 0.0 with all sequences, we don't want that
			Vec2f samplePos;
			if(XY_LP)
			{
				// [0,1) Larcher-Pillichshammer with random scramble.

				samplePos = Vec2f(larcherPillichshammer(j,y*m_width+x),(i+0.5f)/m_numSamplesPerPixel);
			}
			else
			{
				if(XY_HALTON)		samplePos = Vec2f(halton(2,j),halton(3,j));						// [0,1) Halton
				if(XY_HAMMERSLEY)	samplePos = Vec2f(halton(2,j),(i+0.5f)/m_numSamplesPerPixel);	// [0,1) Hammersley
				if(XY_SOBOL)		samplePos = Vec2f(sobol(0,j),sobol(1,j));						// [0,1) Sobol

				// Cranley-Patterson rotations.

				samplePos += offset;
				if(samplePos.x>=1.f) samplePos.x-=1.f;
				if(samplePos.y>=1.f) samplePos.y-=1.f;
			}

			FW_ASSERT(samplePos.x>=0 && samplePos.x<1.f);
			FW_ASSERT(samplePos.y>=0 && samplePos.y<1.f);
			setSampleXY(x,y,i, samplePos + Vec2f(Vec2i(x,y)));
            setSampleWeight(x,y,i, 1.0f);
		}
	}
}

bool SampleBuffer::needRealloc(int w,int h, int numSamplesPerPixel) const
{
	if(w != m_width)								return true;
	if(h != m_height)								return true;
	if(numSamplesPerPixel != m_numSamplesPerPixel)	return true;
	return false;
}

void SampleBuffer::clear(const Vec4f& color,float depth,float w)
{
	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	for(int i=0;i<getNumSamples(x,y);i++)
	{
		setSampleColor(x,y,i, color);
		setSampleDepth(x,y,i, depth);
		setSampleW    (x,y,i, w);
	}
}

void UVTSampleBuffer::clear(const Vec4f& color,float depth,float w)
{
	SampleBuffer::clear(color,depth,w);

	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	for(int i=0;i<getNumSamples(x,y);i++)
	{
		setSampleMV	(x,y,i,0);
		setSampleWG (x,y,i,0);
	}
}

void SampleBuffer::scanOut(Image& image) const
{
	const int w = min(image.getSize().x, m_width );
	const int h = min(image.getSize().y, m_height);

	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	{
		Vec4f outputColor(0.f);
        float weight = 0.f;
		for(int i=0;i<getNumSamples(x,y);i++)
        {
            float K = getSampleWeight(x,y,i);
			outputColor += K*getSampleColor(x,y,i);
            weight += K;
        }
		image.setVec4f( Vec2i(x,y), outputColor / weight );
	}
}

//-------------------------------------------------------------------

UVTSampleBuffer::UVTSampleBuffer(int w,int h, int numSamplesPerPixel)
: SampleBuffer(w,h,numSamplesPerPixel)
{
	Random random(1);

	m_affineMotion		= true;
	m_cocCoeff			= Vec2f(FW_F32_MAX,FW_F32_MAX);
	m_version			= 1.3f;
	m_pixelToFocalPlane = Mat4f();

	m_uv.reset(m_width*m_height*m_numSamplesPerPixel);
	m_t. reset(m_width*m_height*m_numSamplesPerPixel);
	m_mv.reset(m_width*m_height*m_numSamplesPerPixel);
	m_wg.reset(m_width*m_height*m_numSamplesPerPixel);

	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	for(int i=0;i<m_numSamplesPerPixel;i++)
	{
		m_mv[ getIndex(x,y,i) ] = Vec3f(0,0,0);
		m_wg[ getIndex(x,y,i) ] = Vec2f(0,0);
	}

	generateSobolCoop(random);
}

UVTSampleBuffer::UVTSampleBuffer(const char* filename)
{
	m_irregular = false;

	FILE* fp  = fopen(filename, "rb");
	if(!fp)
		fail("File not found");
	FILE* fph = fopen((String(filename)+String(".header")).getPtr(),"rt");
	bool separateHeader = (fph!=NULL);
	if(!separateHeader)
		fph = fp;

	printf("Importing sample buffer... ");

	// Parse version and image size.

	fscanf(fph, "Version %f\n", &m_version);
	fscanf(fph, "Width %d\n", &m_width);
	fscanf(fph, "Height %d\n", &m_height);
	fscanf(fph, "Samples per pixel %d\n", &m_numSamplesPerPixel);

	if(m_version == 1.2f)
	{
		m_affineMotion = true;
		m_cocCoeff     = Vec2f(FW_F32_MAX,FW_F32_MAX);

		char descriptor[1024];
		fscanf(fph, "\n");
		fscanf(fph, "%s\n", &descriptor);

		// Reserve buffers.

		m_xy   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_uv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_t    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_color.reset(m_width*m_height*m_numSamplesPerPixel);
		m_depth.reset(m_width*m_height*m_numSamplesPerPixel);
		m_w    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_mv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_wg   .reset(m_width*m_height*m_numSamplesPerPixel);
		const int CID = reserveChannel<float>("COC");

		// Parse samples.

		printf("\n");
		for(int y=0;y<m_height;y++)
		{
			for(int x=0;x<m_width;x++)
			for(int i=0;i<getNumSamples(x,y);i++)
			{
				const int NUM_ARGS = 14;
				float sx,sy,u,v,t,r,g,b,z,coc_radius,motion_x,motion_y,wgrad_x,wgrad_y;
				int numRead = fscanf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,\n",  &sx,&sy,&u,&v,&t,&r,&g,&b,&z,&coc_radius,&motion_x,&motion_y,&wgrad_x,&wgrad_y);
				if(numRead != NUM_ARGS)
					numRead = fscanf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", &sx,&sy,&u,&v,&t,&r,&g,&b,&z,&coc_radius,&motion_x,&motion_y,&wgrad_x,&wgrad_y);

				FW_ASSERT(numRead == NUM_ARGS);
				FW_ASSERT(sx>=0 && sy>=0 && sx<m_width && sy<m_height);
				FW_ASSERT(u>=-1 && v>=-1 && u<=1 && v<=1);
				FW_ASSERT(t>=0 && t<=1);

				setSampleXY				(x,y,i, Vec2f(sx,sy));
				setSampleUV				(x,y,i, Vec2f(u,v));
				setSampleT 				(x,y,i, t);
				setSampleColor			(x,y,i, Vec4f(r,g,b,1));
				setSampleDepth			(x,y,i, z);
				setSampleW				(x,y,i, z);								// incorrect, but used only for depth sorting in reconstruction
				setSampleExtra<float>	(CID,x,y,i, coc_radius);
				setSampleMV				(x,y,i, Vec3f(motion_x,motion_y,0));	// affine
				setSampleWG				(x,y,i, Vec2f(wgrad_x,wgrad_y));
			}

			printf("%d%%\r", 100*y/m_height);
		}
	}
	else if(m_version == 1.3f)
	{
		// Parse the rest of the header.

		char motionModel[1024];
		fscanf(fph, "Motion model: %s\n", motionModel);
		m_affineMotion = (String(motionModel) == String("affine"));

		fscanf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", &m_cocCoeff[0],&m_cocCoeff[1]);

		bool binary = false;
		char encoding[1024];
		if(fscanf(fph, "Encoding = %s\n", encoding)==1)
			binary = String(encoding) == String("binary");

		fscanf(fph, "\n");

		char descriptor[1024];
		fscanf(fph, "%s\n", descriptor);

		// Reserve buffers.

		m_xy   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_uv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_t    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_color.reset(m_width*m_height*m_numSamplesPerPixel);
		m_depth.reset(m_width*m_height*m_numSamplesPerPixel);
		m_w    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_mv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_wg   .reset(m_width*m_height*m_numSamplesPerPixel);

		// Parse samples.

		Array<Entry13> entries;
		const int num = m_width*m_height*m_numSamplesPerPixel;
		entries.reset(num);

		printf("\n");
		if(binary)
		{
			fread(entries.getPtr(),sizeof(Entry13),num,fp);
		}
		else
		{
			char line[4096];
			int sidx = 0;
			for(int y=0;y<m_height;y++)
			{
				for(int x=0;x<m_width;x++)
				for(int i=0;i<getNumSamples(x,y);i++)
				{
					fgets(line,4096,fp);
					const char* linePtr = line;
					const int NUM_ARGS = sizeof(Entry13)/sizeof(float);
					Entry13& e = entries[sidx++];
					float* vals = (float*)&e;
					int numRead = 0;
					for(int v=0;v<NUM_ARGS;v++)
					{
						parseSpace(linePtr);
						parseChar (linePtr,',');
						parseSpace(linePtr);
						if(parseFloat(linePtr,vals[v]))
							numRead++;
					}

					FW_ASSERT(numRead == NUM_ARGS);
					FW_ASSERT(e.x>=0 && e.y>=0 && e.x<m_width && e.y<m_height);
					FW_ASSERT(e.u>=-1 && e.v>=-1 && e.u<=1 && e.v<=1);
					FW_ASSERT(e.t>=0 && e.t<=1);
				}
				printf("Parsing: %d%%\r", 100*y/m_height);
			}
			printf("                                               \r");
		}
		
		int sidx = 0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			const Entry13& e = entries[sidx++];
			setSampleXY				(x,y,i, Vec2f(e.x,e.y));
			setSampleDepth			(x,y,i, e.z);
			setSampleW				(x,y,i, e.w);
			setSampleUV				(x,y,i, Vec2f(e.u,e.v));
			setSampleT 				(x,y,i, e.t);
			setSampleColor			(x,y,i, Vec4f(e.r,e.g,e.b,1));			// TODO: alpha
			setSampleMV				(x,y,i, Vec3f(e.mv_x,e.mv_y,e.mv_w));
			setSampleWG				(x,y,i, Vec2f(e.dwdx,e.dwdy));
		}
	}
	else if(m_version == 2.0f)
	{
		// Parse the rest of the header.

		m_affineMotion = false;
		
		if(fscanf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", &m_cocCoeff[0],&m_cocCoeff[1])!=2)
			fail("CoC coefficients needs to specify 2 values");

		float* m = (float*)&m_pixelToFocalPlane;
		int n0 = fscanf(fph, "Pixel-to-camera matrix: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", m+0,m+1,m+2,m+3,m+4,m+5,m+6,m+7,m+8,m+9,m+10,m+11,m+12,m+13,m+14,m+15);
		if(n0!=16)
			fail("Pixel-to-camera matrix needs to define 16 values (%d)", n0);

		bool binary = false;
		char encoding[1024];
		if(fscanf(fph, "Encoding = %s\n", encoding)==1)
			binary = String(encoding) == String("binary");

		fscanf(fph, "\n");

		char descriptor[1024];
		fscanf(fph, "%s\n", descriptor);

		// Reserve buffers.

		m_xy   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_uv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_t    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_color.reset(m_width*m_height*m_numSamplesPerPixel);
		m_depth.reset(m_width*m_height*m_numSamplesPerPixel);		// not used
		m_w    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_mv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_wg   .reset(m_width*m_height*m_numSamplesPerPixel);		// not used

		const int CID_PRI_NORMAL   = reserveChannel<Vec3f>(CID_PRI_NORMAL_NAME);
		const int CID_ALBEDO       = reserveChannel<Vec3f>(CID_ALBEDO_NAME);
		const int CID_SEC_ORIGIN   = reserveChannel<Vec3f>(CID_SEC_ORIGIN_NAME);
		const int CID_SEC_HITPOINT = reserveChannel<Vec3f>(CID_SEC_HITPOINT_NAME);
		const int CID_SEC_MV       = reserveChannel<Vec3f>(CID_SEC_MV_NAME);
		const int CID_SEC_NORMAL   = reserveChannel<Vec3f>(CID_SEC_NORMAL_NAME);
		const int CID_DIRECT       = reserveChannel<Vec3f>(CID_DIRECT_NAME);

		// Parse samples.

		Array<Entry20> entries;
		const int num = m_width*m_height*m_numSamplesPerPixel;
		entries.reset(num);

		printf("\n");
		if(binary)
		{
			fread(entries.getPtr(),sizeof(Entry20),num,fp);
		}
		else
		{
			int sidx = 0;
			char line[4096];
			for(int y=0;y<m_height;y++)
			{
				for(int x=0;x<m_width;x++)
				for(int i=0;i<getNumSamples(x,y);i++)
				{
					if(!fgets(line,4096,fp))
						fail("Fileformat 2.0: Buffer contains fewer samples than expected");

					const char* linePtr = line;
					const int NUM_ARGS = sizeof(Entry20)/sizeof(float);
					Entry20& e = entries[sidx++];
					float* vals = (float*)&e;
					int numRead = 0;
					for(int v=0;v<NUM_ARGS;v++)
					{
						parseSpace(linePtr);
						parseChar (linePtr,',');
						parseSpace(linePtr);
						if(parseFloat(linePtr,vals[v]))
							numRead++;
					}

					if(numRead != NUM_ARGS)
						fail("Fileformat 2.0: Wrong number of arguments per line (expected %d, got %d)", NUM_ARGS,numRead);

					FW_ASSERT(e.x>=0 && e.y>=0 && e.x<m_width && e.y<m_height);
					FW_ASSERT(e.u>=-1 && e.v>=-1 && e.u<=1 && e.v<=1);
					FW_ASSERT(e.t>=0 && e.t<=1);
				}
				printf("Parsing: %d%%\r", 100*y/m_height);
			}
			printf("                                               \r");
		}

		int sidx = 0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			const Entry20& e = entries[sidx++];
			setSampleXY				(x,y,i, Vec2f(e.x,e.y));
			setSampleW				(x,y,i, e.w);
			setSampleUV				(x,y,i, Vec2f(e.u,e.v));
			setSampleT 				(x,y,i, e.t);
			setSampleColor			(x,y,i, Vec4f(e.r,e.g,e.b,1));
			setSampleMV				(x,y,i, e.pri_mv);
			setSampleExtra<Vec3f>	(CID_PRI_NORMAL  , x,y,i, e.pri_normal);
			setSampleExtra<Vec3f>	(CID_ALBEDO      , x,y,i, e.albedo);
			setSampleExtra<Vec3f>	(CID_SEC_ORIGIN  , x,y,i, e.sec_origin);
			setSampleExtra<Vec3f>	(CID_SEC_HITPOINT, x,y,i, e.sec_hitpoint);
			setSampleExtra<Vec3f>	(CID_SEC_MV      , x,y,i, e.sec_mv);
			setSampleExtra<Vec3f>	(CID_SEC_NORMAL  , x,y,i, e.sec_normal);
			setSampleExtra<Vec3f>	(CID_DIRECT      , x,y,i, e.direct);
			setSampleDepth			(x,y,i, 0);					// clear unused
			setSampleWG				(x,y,i, Vec2f(0));			// clear unused
		}
	}
	else if(m_version == 2.1f)
	{
		// Parse the rest of the header.

		m_affineMotion = false;
		m_irregular = true;
		
		int numSamples;
		fscanf(fph, "Samples %d\n", &numSamples);
		m_numSamplesPerPixel = -1;	// invalidate

		if(fscanf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", &m_cocCoeff[0],&m_cocCoeff[1])!=2)
			fail("CoC coefficients needs to specify 2 values");

		float* m = (float*)&m_pixelToFocalPlane;
		int n0 = fscanf(fph, "Pixel-to-camera matrix: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", m+0,m+1,m+2,m+3,m+4,m+5,m+6,m+7,m+8,m+9,m+10,m+11,m+12,m+13,m+14,m+15);
		if(n0!=16)
			fail("Pixel-to-camera matrix needs to define 16 values (%d)", n0);

		bool binary = false;
		char encoding[1024];
		if(fscanf(fph, "Encoding = %s\n", encoding)==1)
			binary = String(encoding) == String("binary");

		fscanf(fph, "\n");

		char descriptor[1024];
		fscanf(fph, "%s\n", descriptor);

		// Reserve buffers.

		m_xy   .reset(numSamples);
		m_uv   .reset(numSamples);
		m_t    .reset(numSamples);
		m_color.reset(numSamples);
		m_depth.reset(numSamples);		// not used
		m_w    .reset(numSamples);
		m_mv   .reset(numSamples);
		m_wg   .reset(numSamples);		// not used

		const int CID_PRI_NORMAL   = reserveChannel<Vec3f>(CID_PRI_NORMAL_NAME);
		const int CID_ALBEDO       = reserveChannel<Vec3f>(CID_ALBEDO_NAME);
		const int CID_SEC_ORIGIN   = reserveChannel<Vec3f>(CID_SEC_ORIGIN_NAME);
		const int CID_SEC_HITPOINT = reserveChannel<Vec3f>(CID_SEC_HITPOINT_NAME);
		const int CID_SEC_MV       = reserveChannel<Vec3f>(CID_SEC_MV_NAME);
		const int CID_SEC_NORMAL   = reserveChannel<Vec3f>(CID_SEC_NORMAL_NAME);
		const int CID_DIRECT       = reserveChannel<Vec3f>(CID_DIRECT_NAME);

		m_numSamples .reset(m_width*m_height);	// one per pixel
		m_firstSample.reset(m_width*m_height);	// one per pixel

		// Parse samples.

		Array<Entry21> entries;
		const int num = numSamples;
		entries.reset(num);

		printf("\n");
		if(binary)
		{
			fread(entries.getPtr(),sizeof(Entry21),num,fp);
		}
		else
		{
			int sidx = 0;
			char line[4096];
			for(int i=0;i<numSamples;i++)
			{
				if(!fgets(line,4096,fp))
					fail("Fileformat 2.1: Buffer contains fewer samples than expected");

				const char* linePtr = line;
				const int NUM_ARGS = sizeof(Entry21)/sizeof(float);
				Entry21& e = entries[sidx++];
				float* vals = (float*)&e;
				int numRead = 0;
				for(int v=0;v<NUM_ARGS;v++)
				{
					parseSpace(linePtr);
					parseChar (linePtr,',');
					parseSpace(linePtr);
					if(parseFloat(linePtr,vals[v]))
						numRead++;
				}

				if(numRead != NUM_ARGS)
					fail("Fileformat 2.1: Wrong number of arguments per line (expected %d, got %d)", NUM_ARGS,numRead);

				FW_ASSERT(e.x>=0 && e.y>=0 && e.x<m_width && e.y<m_height);
				FW_ASSERT(e.u>=-1 && e.v>=-1 && e.u<=1 && e.v<=1);
				FW_ASSERT(e.t>=0 && e.t<=1);

				if(i%10000 == 0)
					printf("Parsing: %d%%\r", 100*i/numSamples);
			}
			printf("                                               \r");
		}

		// Count samples in each pixel. We assume samples are provided in pixel-order.

		for(int i=0;i<m_width*m_height;i++)
		{
			m_numSamples [i] = 0;		// clear the arrays because some pixels may not get any samples (yet these fields must be valid)
			m_firstSample[i] = 0;
		}

		Vec2i currentPixel(-1,-1);
		for(int i=0;i<numSamples;i++)
		{
			const Entry21& e = entries[i];
			const Vec2i pixel( int(floor(e.x)),int(floor(e.y)) );
			if(pixel != currentPixel)
			{
				if((pixel.x < currentPixel.x && pixel.y==currentPixel.y) || (pixel.y < currentPixel.y))	// moved left on the same scanline or moved to earlier scanline
					fail("Samples provided in a wrong order (pixel-order required)");
				currentPixel = pixel;
				m_firstSample[pixel.y*m_width+pixel.x] = i;
			}
			m_numSamples[ pixel.y*m_width+pixel.x ]++;
		}

		// Set per-sample data

		int sidx = 0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			const Entry21& e = entries[sidx++];
			setSampleXY				(x,y,i, Vec2f(e.x,e.y));
			setSampleW				(x,y,i, e.w);
			setSampleUV				(x,y,i, Vec2f(e.u,e.v));
			setSampleT 				(x,y,i, e.t);
			setSampleColor			(x,y,i, Vec4f(e.r,e.g,e.b,1));
			setSampleMV				(x,y,i, e.pri_mv);
			setSampleExtra<Vec3f>	(CID_PRI_NORMAL  , x,y,i, e.pri_normal);
			setSampleExtra<Vec3f>	(CID_ALBEDO      , x,y,i, e.albedo);
			setSampleExtra<Vec3f>	(CID_SEC_ORIGIN  , x,y,i, e.sec_origin);
			setSampleExtra<Vec3f>	(CID_SEC_HITPOINT, x,y,i, e.sec_hitpoint);
			setSampleExtra<Vec3f>	(CID_SEC_MV      , x,y,i, e.sec_mv);
			setSampleExtra<Vec3f>	(CID_SEC_NORMAL  , x,y,i, e.sec_normal);
			setSampleExtra<Vec3f>	(CID_DIRECT      , x,y,i, e.direct);
			setSampleDepth			(x,y,i, 0);					// clear unused
			setSampleWG				(x,y,i, Vec2f(0));			// clear unused
			// TODO: add new fields
		}
	}
	else if(m_version == 2.2f)
	{
		// Parse the rest of the header.

		m_affineMotion = false;
		
		if(fscanf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", &m_cocCoeff[0],&m_cocCoeff[1])!=2)
			fail("CoC coefficients needs to specify 2 values");

		float* m = (float*)&m_pixelToFocalPlane;
		int n0 = fscanf(fph, "Pixel-to-camera matrix: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", m+0,m+1,m+2,m+3,m+4,m+5,m+6,m+7,m+8,m+9,m+10,m+11,m+12,m+13,m+14,m+15);
		if(n0!=16)
			fail("Pixel-to-camera matrix needs to define 16 values (%d)", n0);

		bool binary = false;
		char encoding[1024];
		if(fscanf(fph, "Encoding = %s\n", encoding)==1)
			binary = String(encoding) == String("binary");

		fscanf(fph, "\n");

		char descriptor[1024];
		fscanf(fph, "%s\n", descriptor);

		// Reserve buffers.

		m_xy   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_uv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_t    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_color.reset(m_width*m_height*m_numSamplesPerPixel);
		m_depth.reset(m_width*m_height*m_numSamplesPerPixel);		// not used
		m_w    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_mv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_wg   .reset(m_width*m_height*m_numSamplesPerPixel);		// not used

		const int CID_PRI_NORMAL   = reserveChannel<Vec3f>(CID_PRI_NORMAL_NAME);
		const int CID_ALBEDO       = reserveChannel<Vec3f>(CID_ALBEDO_NAME);
		const int CID_SEC_ORIGIN   = reserveChannel<Vec3f>(CID_SEC_ORIGIN_NAME);
		const int CID_SEC_HITPOINT = reserveChannel<Vec3f>(CID_SEC_HITPOINT_NAME);
		const int CID_SEC_MV       = reserveChannel<Vec3f>(CID_SEC_MV_NAME);
		const int CID_SEC_NORMAL   = reserveChannel<Vec3f>(CID_SEC_NORMAL_NAME);
		const int CID_DIRECT       = reserveChannel<Vec3f>(CID_DIRECT_NAME);
		const int CID_SEC_ALBEDO   = reserveChannel<Vec3f>(CID_SEC_ALBEDO_NAME);
		const int CID_SEC_DIRECT   = reserveChannel<Vec3f>(CID_SEC_DIRECT_NAME);

		// Parse samples.

		Array<Entry22> entries;
		const int num = m_width*m_height*m_numSamplesPerPixel;
		entries.reset(num);

		printf("\n");
		if(binary)
		{
			fread(entries.getPtr(),sizeof(Entry22),num,fp);
		}
		else
		{
			int sidx = 0;
			char line[4096];
			for(int y=0;y<m_height;y++)
			{
				for(int x=0;x<m_width;x++)
				for(int i=0;i<getNumSamples(x,y);i++)
				{
					if(!fgets(line,4096,fp))
						fail("Fileformat 2.2: Buffer contains fewer samples than expected");

					const char* linePtr = line;
					const int NUM_ARGS = sizeof(Entry22)/sizeof(float);
					Entry22& e = entries[sidx++];
					float* vals = (float*)&e;
					int numRead = 0;
					for(int v=0;v<NUM_ARGS;v++)
					{
						parseSpace(linePtr);
						parseChar (linePtr,',');
						parseSpace(linePtr);
						if(parseFloat(linePtr,vals[v]))
							numRead++;
					}

					if(numRead != NUM_ARGS)
						fail("Fileformat 2.2: Wrong number of arguments per line (expected %d, got %d)", NUM_ARGS,numRead);

					FW_ASSERT(e.x>=0 && e.y>=0 && e.x<m_width && e.y<m_height);
					FW_ASSERT(e.u>=-1 && e.v>=-1 && e.u<=1 && e.v<=1);
					FW_ASSERT(e.t>=0 && e.t<=1);
				}
				printf("Parsing: %d%%\r", 100*y/m_height);
			}
			printf("                                               \r");
		}

		int sidx = 0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			const Entry22& e = entries[sidx++];
			setSampleXY				(x,y,i, Vec2f(e.x,e.y));
			setSampleW				(x,y,i, e.w);
			setSampleUV				(x,y,i, Vec2f(e.u,e.v));
			setSampleT 				(x,y,i, e.t);
			setSampleColor			(x,y,i, Vec4f(e.r,e.g,e.b,1));
			setSampleMV				(x,y,i, e.pri_mv);
			setSampleExtra<Vec3f>	(CID_PRI_NORMAL  , x,y,i, e.pri_normal);
			setSampleExtra<Vec3f>	(CID_ALBEDO      , x,y,i, e.albedo);
			setSampleExtra<Vec3f>	(CID_SEC_ORIGIN  , x,y,i, e.sec_origin);
			setSampleExtra<Vec3f>	(CID_SEC_HITPOINT, x,y,i, e.sec_hitpoint);
			setSampleExtra<Vec3f>	(CID_SEC_MV      , x,y,i, e.sec_mv);
			setSampleExtra<Vec3f>	(CID_SEC_NORMAL  , x,y,i, e.sec_normal);
			setSampleExtra<Vec3f>	(CID_DIRECT      , x,y,i, e.direct);
			setSampleExtra<Vec3f>	(CID_SEC_ALBEDO  , x,y,i, e.sec_albedo);
			setSampleExtra<Vec3f>	(CID_SEC_DIRECT  , x,y,i, e.sec_direct);
			setSampleDepth			(x,y,i, 0);					// clear unused
			setSampleWG				(x,y,i, Vec2f(0));			// clear unused
		}
	}
	else
		fail("Unsupported sample stream version (%.1f)", m_version);

	fclose(fp);
	if(separateHeader)
		fclose(fph);
	printf("done\n");
}

void UVTSampleBuffer::serialize(const char* filename, bool separateHeader, bool binary) const
{
	if(binary && !separateHeader)
		fail("binary serialization supported only with a separate header");

	FILE* fp = (binary) ? fopen(filename, "wb") : fopen(filename, "wt");
	FILE* fph= (separateHeader) ? fopen((String(filename)+String(".header")).getPtr(),"wt") : fp;

	printf("Serializing sample buffer... ");

	if(m_version==1.2f)
	{
		// header
		fprintf(fph, "Version 1.2\n");
		fprintf(fph, "Width %d\n", m_width);
		fprintf(fph, "Height %d\n", m_height);
		fprintf(fph, "Samples per pixel %d\n", m_numSamplesPerPixel);
		fprintf(fph, "\n");
		fprintf(fph, "x,y,u,v,t,r,g,b,z,coc_radius,motion_x,motion_y,wgrad_x,wgrad_y\n");

		const int CID = getChannelID("COC");

		// samples
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			fprintf(fp, "%f,", getSampleXY(x,y,i)[0]);			// x
			fprintf(fp, "%f,", getSampleXY(x,y,i)[1]);			// y
			fprintf(fp, "%f,", getSampleUV(x,y,i)[0]);			// u
			fprintf(fp, "%f,", getSampleUV(x,y,i)[1]);			// v
			fprintf(fp, "%f,", getSampleT (x,y,i));				// t

			fprintf(fp, "%f,", getSampleColor(x,y,i)[0]);		// r
			fprintf(fp, "%f,", getSampleColor(x,y,i)[1]);		// g
			fprintf(fp, "%f,", getSampleColor(x,y,i)[2]);		// b
			fprintf(fp, "%f,", getSampleDepth(x,y,i));			// z

			fprintf(fp, "%f,", getSampleFloat(CID,x,y,i));		// coc radius
			fprintf(fp, "%f,", getSampleMV(x,y,i)[0]);			// motion vector.x
			fprintf(fp, "%f,", getSampleMV(x,y,i)[1]);			// motion vector.y
			fprintf(fp, "%f,", getSampleWG(x,y,i)[0]);			// w gradient.x
			fprintf(fp, "%f",  getSampleWG(x,y,i)[1]);			// w gradient.y
			fprintf(fp, "\n"); 
		}
	} // 1.2

	else if(m_version==1.3f)
	{
		if(m_cocCoeff == Vec2f(FW_F32_MAX,FW_F32_MAX))
			fail("coc coefficients not set");

		// header
		fprintf(fph, "Version 1.3\n");
		fprintf(fph, "Width %d\n", m_width);
		fprintf(fph, "Height %d\n", m_height);
		fprintf(fph, "Samples per pixel %d\n", m_numSamplesPerPixel);
		fprintf(fph, "Motion model: %s\n", m_affineMotion ? "affine" : "perspective");
		fprintf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", m_cocCoeff[0], m_cocCoeff[1]);
		fprintf(fph, "Encoding = %s\n", binary ? "binary" : "text");
		fprintf(fph, "x,y,z/w,w,u,v,t,r,g,b,a,mv_x,mv_y,mv_w,dwdx,dwdy\n");

		// Using Wikipedia's terminology
		//
		//    c = AD * (objectDist-focusDist)/objectDist * f/(focusDist-f)
		// => c = AD * f/(focusDist-f) * (1-focusDist/objectDist)
		// => c = C1 * (1-focusDist/objectDist)
		// => c = C1 - C1*focusDist/objectDist
		// => c = C1 + C0/objectDist
		//
		// C1 = ApertureDiameter * f/(focusDist-f)
		// C0 = -C1*focusDist

		int num = 0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width; x++)
			num += getNumSamples(x,y);

		Array<Entry13> entries;
		entries.reset(num);

		int sidx=0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			Entry13& e = entries[sidx++];
			e.x = getSampleXY(x,y,i)[0];			// x	(in window coordinates, NOT multiplied with w)
			e.y = getSampleXY(x,y,i)[1];			// y	(in window coordinates, NOT multiplied with w)
			e.z = getSampleDepth(x,y,i);			// z	(z/w as in OpenGL, not used by reconstruction)
			e.w = getSampleW(x,y,i);				// w	(camera-space z. Positive are visible, larger is farther).

			e.u = getSampleUV(x,y,i)[0];			// u	[-1,1]
			e.v = getSampleUV(x,y,i)[1];			// v	[-1,1]
			e.t = getSampleT (x,y,i);				// t	[0,1]

			e.r = getSampleColor(x,y,i)[0];			// r	[0,1]
			e.g = getSampleColor(x,y,i)[1];			// g	[0,1]
			e.b = getSampleColor(x,y,i)[2];			// b	[0,1]
			e.a = getSampleColor(x,y,i)[2];			// a	[0,1]				TODO

			e.mv_x = getSampleMV(x,y,i)[0];			// homogeneous motion vector.x
			e.mv_y = getSampleMV(x,y,i)[1];			// homogeneous motion vector.y
			e.mv_w = getSampleMV(x,y,i)[2];			// homogeneous motion vector.w

			e.dwdx = getSampleWG(x,y,i)[0];			// dw/dx
			e.dwdy = getSampleWG(x,y,i)[1];			// dw/dy
		}

		if(binary)
		{
			fwrite(entries.getPtr(), sizeof(Entry13), num, fp);
		}
		else
		{
			printf("\n");
			for(int i=0;i<entries.getSize();i++)
			{
				Entry13& e = entries[i];
				float* vals = (float*)&e;
				for(int j=0;j<sizeof(Entry13)/sizeof(float);j++)
					fprintf(fp, "%f,", vals[j]);
				fprintf(fp, "\n"); 
				if(i%10000 == 0)
					printf("Writing to file: %d%%\r", 100*i/entries.getSize());
			}
			printf("                                               \rdone\n");
		}
	} // 1.3
	else if(m_version == 2.f)
	{
		// get IDs of extra channels
		const int CID_PRI_NORMAL   = getChannelID(CID_PRI_NORMAL_NAME  );
		const int CID_ALBEDO       = getChannelID(CID_ALBEDO_NAME      );
		const int CID_SEC_ORIGIN   = getChannelID(CID_SEC_ORIGIN_NAME  );
		const int CID_SEC_HITPOINT = getChannelID(CID_SEC_HITPOINT_NAME);
		const int CID_SEC_MV       = getChannelID(CID_SEC_MV_NAME      );
		const int CID_SEC_NORMAL   = getChannelID(CID_SEC_NORMAL_NAME  );
		const int CID_DIRECT       = getChannelID(CID_DIRECT_NAME      );

		if(CID_PRI_NORMAL  ==-1)	fail("Serialize: channel %s not defined", CID_PRI_NORMAL_NAME  );
		if(CID_ALBEDO      ==-1)	fail("Serialize: channel %s not defined", CID_ALBEDO_NAME      );
		if(CID_SEC_ORIGIN  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_ORIGIN_NAME  );
		if(CID_SEC_HITPOINT==-1)	fail("Serialize: channel %s not defined", CID_SEC_HITPOINT_NAME);
		if(CID_SEC_MV      ==-1)	fail("Serialize: channel %s not defined", CID_SEC_MV_NAME      );
		if(CID_SEC_NORMAL  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_NORMAL_NAME  );
		if(CID_DIRECT      ==-1)	fail("Serialize: channel %s not defined", CID_DIRECT_NAME      );

		// header
		if(m_cocCoeff == Vec2f(FW_F32_MAX,FW_F32_MAX))
			fail("coc coefficients not set");
		float* m = (float*)&m_pixelToFocalPlane;

		fprintf(fph, "Version 2.0\n");
		fprintf(fph, "Width %d\n", m_width);
		fprintf(fph, "Height %d\n", m_height);
		fprintf(fph, "Samples per pixel %d\n", m_numSamplesPerPixel);
		fprintf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", m_cocCoeff[0], m_cocCoeff[1]);
		fprintf(fph, "Pixel-to-camera matrix: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", *(m+0),*(m+1),*(m+2),*(m+3),*(m+4),*(m+5),*(m+6),*(m+7),*(m+8),*(m+9),*(m+10),*(m+11),*(m+12),*(m+13),*(m+14),*(m+15));
		fprintf(fph, "Encoding = %s\n", binary ? "binary" : "text");
		fprintf(fph, "x,y,w,u,v,t,r,g,b,%s(3d),%s(3d),%s,%s(3d),%s(3d),%s(3d),%s\n", CID_PRI_MV_NAME,CID_PRI_NORMAL_NAME,CID_ALBEDO_NAME,CID_SEC_ORIGIN_NAME,CID_SEC_HITPOINT_NAME,CID_SEC_MV_NAME,CID_SEC_NORMAL_NAME,CID_DIRECT_NAME);

		int num = 0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width; x++)
			num += getNumSamples(x,y);

		Array<Entry20> entries;
		entries.reset(num);

		int sidx=0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			Entry20& e = entries[sidx++];
			e.x = getSampleXY(x,y,i)[0];			// x	(in window coordinates, NOT multiplied with w)
			e.y = getSampleXY(x,y,i)[1];			// y	(in window coordinates, NOT multiplied with w)
			e.w = getSampleW(x,y,i);				// w	(camera-space z. Positive are visible, larger is farther).

			e.u = getSampleUV(x,y,i)[0];			// u	[-1,1]
			e.v = getSampleUV(x,y,i)[1];			// v	[-1,1]
			e.t = getSampleT (x,y,i);				// t	[0,1]

			e.r = getSampleColor(x,y,i)[0];			// r	[0,1]
			e.g = getSampleColor(x,y,i)[1];			// g	[0,1]
			e.b = getSampleColor(x,y,i)[2];			// b	[0,1]

			e.pri_mv	   = getSampleMV(x,y,i);
			e.pri_normal   = getSampleExtra<Vec3f>(CID_PRI_NORMAL  ,x,y,i);		// primitive normal (3D)
			e.albedo       = getSampleExtra<Vec3f>(CID_ALBEDO      ,x,y,i);		// primary hit albedo
			e.sec_origin   = getSampleExtra<Vec3f>(CID_SEC_ORIGIN  ,x,y,i);		// secondary ray origin (3D)
			e.sec_hitpoint = getSampleExtra<Vec3f>(CID_SEC_HITPOINT,x,y,i);		// secondary ray hit point (3D)
			e.sec_mv       = getSampleExtra<Vec3f>(CID_SEC_MV      ,x,y,i);		// secondary ray hit point motion vector (3D)
			e.sec_normal   = getSampleExtra<Vec3f>(CID_SEC_NORMAL  ,x,y,i);		// secondary ray hit normal (3D)
			e.direct       = getSampleExtra<Vec3f>(CID_DIRECT      ,x,y,i);		// direct light (r,g,b)
		}

		if(binary)
		{
			fwrite(entries.getPtr(), sizeof(Entry20), num, fp);
		}
		else
		{
			printf("\n");
			for(int i=0;i<entries.getSize();i++)
			{
				Entry20& e = entries[i];
				float* vals = (float*)&e;
				for(int j=0;j<sizeof(Entry20)/sizeof(float);j++)
					fprintf(fp, "%f,", vals[j]);
				fprintf(fp, "\n"); 
				if(i%10000 == 0)
					printf("Writing to file: %d%%\r", 100*i/entries.getSize());
			}
			printf("                                               \rdone\n");
		}
	}
	else if(m_version == 2.1f)
	{
		// get IDs of extra channels
		const int CID_PRI_NORMAL   = getChannelID(CID_PRI_NORMAL_NAME  );
		const int CID_ALBEDO       = getChannelID(CID_ALBEDO_NAME      );
		const int CID_SEC_ORIGIN   = getChannelID(CID_SEC_ORIGIN_NAME  );
		const int CID_SEC_HITPOINT = getChannelID(CID_SEC_HITPOINT_NAME);
		const int CID_SEC_MV       = getChannelID(CID_SEC_MV_NAME      );
		const int CID_SEC_NORMAL   = getChannelID(CID_SEC_NORMAL_NAME  );
		const int CID_DIRECT       = getChannelID(CID_DIRECT_NAME      );

		if(CID_PRI_NORMAL  ==-1)	fail("Serialize: channel %s not defined", CID_PRI_NORMAL_NAME  );
		if(CID_ALBEDO      ==-1)	fail("Serialize: channel %s not defined", CID_ALBEDO_NAME      );
		if(CID_SEC_ORIGIN  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_ORIGIN_NAME  );
		if(CID_SEC_HITPOINT==-1)	fail("Serialize: channel %s not defined", CID_SEC_HITPOINT_NAME);
		if(CID_SEC_MV      ==-1)	fail("Serialize: channel %s not defined", CID_SEC_MV_NAME      );
		if(CID_SEC_NORMAL  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_NORMAL_NAME  );
		if(CID_DIRECT      ==-1)	fail("Serialize: channel %s not defined", CID_DIRECT_NAME      );
		
		// TODO: new fields

		// header
		if(m_cocCoeff == Vec2f(FW_F32_MAX,FW_F32_MAX))
			fail("coc coefficients not set");
		float* m = (float*)&m_pixelToFocalPlane;

		const int num = m_xy.getSize();

		fprintf(fph, "Version 2.0\n");
		fprintf(fph, "Width %d\n", m_width);
		fprintf(fph, "Height %d\n", m_height);
		fprintf(fph, "Samples per pixel %d\n", -1);
		fprintf(fph, "Samples %d\n", num);
		fprintf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", m_cocCoeff[0], m_cocCoeff[1]);
		fprintf(fph, "Pixel-to-camera matrix: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", *(m+0),*(m+1),*(m+2),*(m+3),*(m+4),*(m+5),*(m+6),*(m+7),*(m+8),*(m+9),*(m+10),*(m+11),*(m+12),*(m+13),*(m+14),*(m+15));
		fprintf(fph, "Encoding = %s\n", binary ? "binary" : "text");
		fprintf(fph, "x,y,w,u,v,t,r,g,b,%s(3d),%s(3d),%s,%s(3d),%s(3d),%s(3d),%s\n", CID_PRI_MV_NAME,CID_PRI_NORMAL_NAME,CID_ALBEDO_NAME,CID_SEC_ORIGIN_NAME,CID_SEC_HITPOINT_NAME,CID_SEC_MV_NAME,CID_SEC_NORMAL_NAME,CID_DIRECT_NAME);
		// TODO: new fields

		Array<Entry21> entries;
		entries.reset(num);

		int sidx=0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			Entry21& e = entries[sidx++];
			e.x = getSampleXY(x,y,i)[0];			// x	(in window coordinates, NOT multiplied with w)
			e.y = getSampleXY(x,y,i)[1];			// y	(in window coordinates, NOT multiplied with w)
			e.w = getSampleW(x,y,i);				// w	(camera-space z. Positive are visible, larger is farther).

			e.u = getSampleUV(x,y,i)[0];			// u	[-1,1]
			e.v = getSampleUV(x,y,i)[1];			// v	[-1,1]
			e.t = getSampleT (x,y,i);				// t	[0,1]

			e.r = getSampleColor(x,y,i)[0];			// r	[0,1]
			e.g = getSampleColor(x,y,i)[1];			// g	[0,1]
			e.b = getSampleColor(x,y,i)[2];			// b	[0,1]

			e.pri_mv	   = getSampleMV(x,y,i);
			e.pri_normal   = getSampleExtra<Vec3f>(CID_PRI_NORMAL  ,x,y,i);		// primitive normal (3D)
			e.albedo       = getSampleExtra<Vec3f>(CID_ALBEDO      ,x,y,i);		// primary hit albedo
			e.sec_origin   = getSampleExtra<Vec3f>(CID_SEC_ORIGIN  ,x,y,i);		// secondary ray origin (3D)
			e.sec_hitpoint = getSampleExtra<Vec3f>(CID_SEC_HITPOINT,x,y,i);		// secondary ray hit point (3D)
			e.sec_mv       = getSampleExtra<Vec3f>(CID_SEC_MV      ,x,y,i);		// secondary ray hit point motion vector (3D)
			e.sec_normal   = getSampleExtra<Vec3f>(CID_SEC_NORMAL  ,x,y,i);		// secondary ray hit normal (3D)
			e.direct       = getSampleExtra<Vec3f>(CID_DIRECT      ,x,y,i);		// direct light (r,g,b)

			// TODO: new fields
		}

		if(binary)
		{
			fwrite(entries.getPtr(), sizeof(Entry21), num, fp);
		}
		else
		{
			printf("\n");
			for(int i=0;i<entries.getSize();i++)
			{
				Entry21& e = entries[i];
				float* vals = (float*)&e;
				for(int j=0;j<sizeof(Entry21)/sizeof(float);j++)
					fprintf(fp, "%f,", vals[j]);
				fprintf(fp, "\n"); 
				if(i%10000 == 0)
					printf("Writing to file: %d%%\r", 100*i/entries.getSize());
			}
			printf("                                               \rdone\n");
		}
	}
	else if(m_version == 2.2f)
	{
		// get IDs of extra channels
		const int CID_PRI_NORMAL   = getChannelID(CID_PRI_NORMAL_NAME  );
		const int CID_ALBEDO       = getChannelID(CID_ALBEDO_NAME      );
		const int CID_SEC_ORIGIN   = getChannelID(CID_SEC_ORIGIN_NAME  );
		const int CID_SEC_HITPOINT = getChannelID(CID_SEC_HITPOINT_NAME);
		const int CID_SEC_MV       = getChannelID(CID_SEC_MV_NAME      );
		const int CID_SEC_NORMAL   = getChannelID(CID_SEC_NORMAL_NAME  );
		const int CID_DIRECT       = getChannelID(CID_DIRECT_NAME      );
		const int CID_SEC_ALBEDO   = getChannelID(CID_SEC_ALBEDO_NAME  );
		const int CID_SEC_DIRECT   = getChannelID(CID_SEC_DIRECT_NAME  );

		if(CID_PRI_NORMAL  ==-1)	fail("Serialize: channel %s not defined", CID_PRI_NORMAL_NAME  );
		if(CID_ALBEDO      ==-1)	fail("Serialize: channel %s not defined", CID_ALBEDO_NAME      );
		if(CID_SEC_ORIGIN  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_ORIGIN_NAME  );
		if(CID_SEC_HITPOINT==-1)	fail("Serialize: channel %s not defined", CID_SEC_HITPOINT_NAME);
		if(CID_SEC_MV      ==-1)	fail("Serialize: channel %s not defined", CID_SEC_MV_NAME      );
		if(CID_SEC_NORMAL  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_NORMAL_NAME  );
		if(CID_DIRECT      ==-1)	fail("Serialize: channel %s not defined", CID_DIRECT_NAME      );
		if(CID_SEC_ALBEDO  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_ALBEDO_NAME  );
		if(CID_SEC_DIRECT  ==-1)	fail("Serialize: channel %s not defined", CID_SEC_DIRECT_NAME  );
		
		// TODO: new fields

		// header
		if(m_cocCoeff == Vec2f(FW_F32_MAX,FW_F32_MAX))
			fail("coc coefficients not set");
		float* m = (float*)&m_pixelToFocalPlane;

		const int num = m_xy.getSize();

		fprintf(fph, "Version 2.2\n");
		fprintf(fph, "Width %d\n", m_width);
		fprintf(fph, "Height %d\n", m_height);
		fprintf(fph, "Samples per pixel %d\n", m_numSamplesPerPixel);
		fprintf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", m_cocCoeff[0], m_cocCoeff[1]);
		fprintf(fph, "Pixel-to-camera matrix: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", *(m+0),*(m+1),*(m+2),*(m+3),*(m+4),*(m+5),*(m+6),*(m+7),*(m+8),*(m+9),*(m+10),*(m+11),*(m+12),*(m+13),*(m+14),*(m+15));
		fprintf(fph, "Encoding = %s\n", binary ? "binary" : "text");
		fprintf(fph, "x,y,w,u,v,t,r,g,b,%s(3d),%s(3d),%s(3d),%s(3d),%s(3d),%s(3d),%s(3d),%s(3d),%s(3d),%s(3d)\n", CID_PRI_MV_NAME,CID_PRI_NORMAL_NAME,CID_ALBEDO_NAME,CID_SEC_ORIGIN_NAME,CID_SEC_HITPOINT_NAME,CID_SEC_MV_NAME,CID_SEC_NORMAL_NAME,CID_DIRECT_NAME,CID_SEC_ALBEDO_NAME,CID_SEC_DIRECT_NAME);
		// TODO: new fields

		Array<Entry22> entries;
		entries.reset(num);

		int sidx=0;
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			Entry22& e = entries[sidx++];
			e.x = getSampleXY(x,y,i)[0];			// x	(in window coordinates, NOT multiplied with w)
			e.y = getSampleXY(x,y,i)[1];			// y	(in window coordinates, NOT multiplied with w)
			e.w = getSampleW(x,y,i);				// w	(camera-space z. Positive are visible, larger is farther).

			e.u = getSampleUV(x,y,i)[0];			// u	[-1,1]
			e.v = getSampleUV(x,y,i)[1];			// v	[-1,1]
			e.t = getSampleT (x,y,i);				// t	[0,1]

			e.r = getSampleColor(x,y,i)[0];			// r	[0,1]
			e.g = getSampleColor(x,y,i)[1];			// g	[0,1]
			e.b = getSampleColor(x,y,i)[2];			// b	[0,1]

			e.pri_mv	   = getSampleMV(x,y,i);
			e.pri_normal   = getSampleExtra<Vec3f>(CID_PRI_NORMAL  ,x,y,i);		// primitive normal (3D)
			e.albedo       = getSampleExtra<Vec3f>(CID_ALBEDO      ,x,y,i);		// primary hit albedo
			e.sec_origin   = getSampleExtra<Vec3f>(CID_SEC_ORIGIN  ,x,y,i);		// secondary ray origin (3D)
			e.sec_hitpoint = getSampleExtra<Vec3f>(CID_SEC_HITPOINT,x,y,i);		// secondary ray hit point (3D)
			e.sec_mv       = getSampleExtra<Vec3f>(CID_SEC_MV      ,x,y,i);		// secondary ray hit point motion vector (3D)
			e.sec_normal   = getSampleExtra<Vec3f>(CID_SEC_NORMAL  ,x,y,i);		// secondary ray hit normal (3D)
			e.direct       = getSampleExtra<Vec3f>(CID_DIRECT      ,x,y,i);		// direct light (r,g,b)
			e.sec_albedo   = getSampleExtra<Vec3f>(CID_SEC_ALBEDO  ,x,y,i);		// direct light (r,g,b)
			e.sec_direct   = getSampleExtra<Vec3f>(CID_SEC_DIRECT  ,x,y,i);		// direct light (r,g,b)

			// TODO: new fields
		}

		if(binary)
		{
			fwrite(entries.getPtr(), sizeof(Entry22), num, fp);
		}
		else
		{
			printf("\n");
			for(int i=0;i<entries.getSize();i++)
			{
				Entry22& e = entries[i];
				float* vals = (float*)&e;
				for(int j=0;j<sizeof(Entry21)/sizeof(float);j++)
					fprintf(fp, "%f,", vals[j]);
				fprintf(fp, "\n"); 
				if(i%10000 == 0)
					printf("Writing to file: %d%%\r", 100*i/entries.getSize());
			}
			printf("                                               \rdone\n");
		}
	}
	else
		fail("serialize -- don't know how to export V%.1f", m_version);


	fflush(fp);
	fclose(fp);
	if(separateHeader)
	{
		fflush(fph);
		fclose(fph);
	}
	printf("done\n");
}

//-------------------------------------------------------------------

void UVTSampleBuffer::generateSobolCoop(Random& random)
{
    Array<Vec4i> shuffle;
    shuffle.reset(24151); // prime
    for (int i = 0; i < shuffle.getSize(); i++)
    {
        shuffle[i] = Vec4i(0, 1, 2, 3);
        for (int j = 4; j >= 2; j--)
            swap(shuffle[i][j - 1], shuffle[i][random.getS32(j)]);
    }

    int sampleIdx = 0;
    for (int py = 0; py < m_height; py++)
    for (int px = 0; px < m_width; px++)
    {
        int morton = 0;
        for (int i = 10; i >= 0; i--)
        {
            int childIdx = ((px >> i) & 1) + ((py >> i) & 1) * 2;
            morton = morton * 4 + shuffle[morton % shuffle.getSize()][childIdx];
        }

        for (int i = 0; i < m_numSamplesPerPixel; i++)
        {
            int j = i + morton * m_numSamplesPerPixel;
            float x = sobol(3, j);
            float y = sobol(4, j);
            float u = sobol(0, j);
            float v = sobol(2, j);
            float t = sobol(1, j);

            m_xy[sampleIdx] = Vec2f(px + x, py + y);
            m_uv[sampleIdx] = ToUnitDisk(Vec2f(u, v));
            m_t[sampleIdx] = t;
            sampleIdx++;
        }
    }
}

//-------------------------------------------------------------------

} //
