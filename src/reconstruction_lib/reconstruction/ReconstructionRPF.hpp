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
Implementation of random parameter filtering [Sen et al. 2012].

- NOTE: Random params added directly to the header, there is no API for this. Search NEW_RANDOM_PARAM to identify all places where a new parameter should be dealt with.
		Similarly for new scene features.
*/
#include "Reconstruction.hpp"

namespace FW
{

static int CID_PRI_MV      ;
static int CID_PRI_NORMAL  ;
static int CID_ALBEDO      ;
static int CID_SEC_ORIGIN  ;
static int CID_SEC_HITPOINT;
static int CID_SEC_MV      ;
static int CID_SEC_NORMAL  ;
static int CID_DIRECT      ;
static int CID_SEC_ALBEDO  ;
static int CID_SEC_DIRECT  ;

//-----------------------------------------------------------------------
// 
//-----------------------------------------------------------------------

#define ACCESSOR(X)	static int	 getSize()					{ return sizeof(X)/sizeof(float); }\
					void   operator+=(const X& s)			{ for(int i=0;i<getSize();i++) (*this)[i] += s[i]; } \
					void   divide(int s)					{ for(int i=0;i<getSize();i++) (*this)[i] /= float(s); } \
					float& operator[](int i)				{ return ((float*)this)[i]; } \
					const float& operator[](int i) const	{ return ((float*)this)[i]; } \
					X(float f=0.f)							{ set(f); } \
					void  set(float f)						{ for(int i=0;i<getSize();i++) (*this)[i] = f; } \
					float sum() const						{ float s=0; for(int i=0;i<getSize();i++) s+=(*this)[i]; return s; } \
					float avg() const						{ return sum()/getSize(); } \
					float min() const						{ float s=FW_F32_MAX;  for(int i=0;i<getSize();i++) s=FW::min(s,(*this)[i]); return s; } \
					float max() const						{ float s=-FW_F32_MAX; for(int i=0;i<getSize();i++) s=FW::max(s,(*this)[i]); return s; }

struct ScreenPosition
{
	UnalignedVec2f xy;
	void fetch(const UVTSampleBuffer& sbuf,int x,int y,int i)	{ xy=sbuf.getSampleXY(x,y,i); }
	ACCESSOR(ScreenPosition);
};

struct ColorFeatures
{
	UnalignedVec3f rgb;
	void fetch(const UVTSampleBuffer& sbuf,int x,int y,int i)	{ rgb=sbuf.getSampleColor(x,y,i).getXYZ(); }
	ACCESSOR(ColorFeatures);
};

struct RandomParams
{
//	UnalignedVec2f uv;		// lens position		NEW_RANDOM_PARAM
//	float t;				// time
	UnalignedVec3f dir;		// first reflection direction (for path tracing)
	void fetch(const UVTSampleBuffer& sbuf,int x,int y,int i)
	{
//		uv  = sbuf.getSampleUV(x,y,i);			//	NEW_RANDOM_PARAM
//		t   = sbuf.getSampleT(x,y,i);
		if(CID_SEC_ORIGIN!=-1 && CID_SEC_HITPOINT!=-1)
			dir = (sbuf.getSampleExtra<Vec3f>(CID_SEC_HITPOINT,x,y,i) - sbuf.getSampleExtra<Vec3f>(CID_SEC_ORIGIN,x,y,i)).normalized();
		else
			dir = Vec3f(0.f);
	}
	ACCESSOR(RandomParams);
};

const int FIRST_NON_POSITION_FEATURES = 6;	// NOTE: different stddev (30 vs. 3) applied to world-space positions during clustering
struct SceneFeatures
{
	UnalignedVec3f p1;		// primary hit point
	UnalignedVec3f p2;		// secondary hit point
	UnalignedVec3f n1;		// primary normal
	UnalignedVec3f n2;		// secondary normal
	UnalignedVec3f albedo;	// albedo (Sen uses "texture values")

	void fetch(const UVTSampleBuffer& sbuf,int x,int y,int i)
	{
		if(CID_SEC_ORIGIN!=-1 && CID_SEC_HITPOINT!=-1 && CID_PRI_NORMAL!=-1 && CID_SEC_NORMAL!=-1 && CID_ALBEDO!=-1)
		{
			p1 = sbuf.getSampleExtra<Vec3f>(CID_SEC_ORIGIN,x,y,i);
			p2 = sbuf.getSampleExtra<Vec3f>(CID_SEC_HITPOINT,x,y,i);
			n1 = sbuf.getSampleExtra<Vec3f>(CID_PRI_NORMAL,x,y,i);
			n2 = sbuf.getSampleExtra<Vec3f>(CID_SEC_NORMAL,x,y,i);
			albedo = sbuf.getSampleExtra<Vec3f>(CID_ALBEDO,x,y,i);
		}
		else
			p1=p2=n1=n2=albedo = Vec3f(0);
	}
	bool isValid() const	{ return p1.max()<1e10f && p2.max()<1e10f; }	// sbuf may have invalid samples
	ACCESSOR(SceneFeatures);
};

struct SampleVector		// Do not reorder the members, some debug functionality relies on this layout.
{
	ScreenPosition	p;
	RandomParams	r;
	SceneFeatures	f;
	ColorFeatures	c;
	void fetch(const UVTSampleBuffer& sbuf,int x,int y,int i)	{ p.fetch(sbuf,x,y,i); r.fetch(sbuf,x,y,i); f.fetch(sbuf,x,y,i); c.fetch(sbuf,x,y,i); }
	ACCESSOR(SampleVector);
};

//-----------------------------------------------------------------------
// Core functionality, data shared among all tasks.
//-----------------------------------------------------------------------

class RPF
{
public:
			RPF					(Image* resultImage, Image* debugImage, const UVTSampleBuffer& sbuf, float aoLength=0.f);

	void	filter				(int iteration);

	const SampleVector&	getUnNormalizedSampleVector(int x,int y,int i) const	{ return m_unNormalizedSamples[getSampleIndex(x,y,i)]; }
	const SampleVector&	getUnNormalizedSampleVector(int index) const			{ return m_unNormalizedSamples[index]; }

	const Vec3f&		getInputColor(int index) const							{ return m_inputColors[index]; }

	Vec3f*				getOutputColorPtr(int index)							{ return &m_outputColors[index]; }

private:
	friend class RPFTask;

	int					getSampleIndex(int x,int y,int i) const					{ return (y*m_w+x)*m_spp+i; }

	Image*					m_image;
	Image*					m_debugImage;

	int						m_w;
	int						m_h;
	int						m_spp;

	Array<SampleVector>		m_unNormalizedSamples;			// "raw" sample data, normalized separately for each environment
	Array<Vec3f>			m_inputColors;					// current iteration uses these colors
	Array<Vec3f>			m_outputColors;					// ... and writes these colors
};

//-----------------------------------------------------------------------
// Multi-Core task
//-----------------------------------------------------------------------

class Random2D;
class RPFTask
{
public:
	void	init	(const RPF* rpf,int t)					{ m_scope = rpf; m_iteration=t; }

	static	void	filter	(MulticoreLauncher::Task& task) { RPFTask* fttask = (RPFTask*)task.data; fttask->filter(task.idx); }
			void	filter	(int y);

private:
	void	getPixelMeanAndStddev		(const Vec2i& pixel, SceneFeatures& E, SceneFeatures& stddev) const;
	void	determineNeighborhood		(Array<int>& neighborIndices, const Vec2i& pixel,int b,int M, Random2D& random) const;
	Vec3f	computeWeights				(ColorFeatures& alpha,SceneFeatures& beta,float& W_r_c, const Array<SampleVector>& normalizedSamples, int iteration);
	Vec4f 	filterColorSamples			(const ColorFeatures& alpha,const SceneFeatures& beta,float W_r_c, const Array<SampleVector>& normalizedSamples,const Array<int>& neighborIndices);

	void	printAllWeights				(const Array<SampleVector>& normalizedSamples) const;					// DEBUG function

	const RPF*				m_scope;		// container for shared data
	int						m_iteration;	// which iteration?

	RandomParams			m_D_rk_c;		// kth random parameter vs. ALL color channels
	ScreenPosition			m_D_pk_c;		// kth screen position vs. ALL color channels
	SceneFeatures			m_D_fk_c;		// kth scene feature vs. ALL color channels

	Array<RandomParams>		m_D_fk_rl;		// kth scene feature vs. lth random parameter
	Array<ScreenPosition>	m_D_fk_pl;		// kth scene feature vs. lth screen position
	Array<ColorFeatures>	m_D_fk_cl;		// kth scene feature vs. lth color feature
};

//-----------------------------------------------------------------------
// Convenience kludge
//-----------------------------------------------------------------------

class SOAAccessor
{
public:
	template<class S>
	SOAAccessor(const Array<S>& A, int offsetInFloats)
	{
		m_ptr	 = (float*)A.getPtr() + offsetInFloats;
		m_size	 = A.getSize();							// #elements in the array
		m_stride = sizeof(S)/sizeof(float);				// #floats per element
	}

		  float*	getPtr()			{ return (float*)m_ptr; }	// TODO: !!!
	const float*	getPtr() const		{ return m_ptr; }
	int				getSize() const		{ return m_size; }			// number of elements in the array
	int				getStride() const	{ return m_stride; }		// number of _floats_ per element

private:
	const float*	m_ptr;
	int				m_size;
	int				m_stride;
};

//-----------------------------------------------------------------------
// Mutual info
//-----------------------------------------------------------------------

template<int N>
class MutualInformation
{
public:
	MutualInformation(bool normalized=false)
	{
		m_numSamples = 0;
		m_histogramA.resize(N);
		m_histogramB.resize(N);
		m_histogramAB.reset(N*N);
		m_normalized = normalized;
	}

	float mutualInformation(const SOAAccessor& A,const SOAAccessor& B)		{ FW_ASSERT(A.getSize() == B.getSize()); return mutualInformation(A.getPtr(),B.getPtr(), A.getSize(),A.getStride()); }
	float mutualInformation(const Array<float>& A,const Array<float>& B)	{ FW_ASSERT(A.getSize() == B.getSize()); return mutualInformation(A.getPtr(),B.getPtr(), A.getSize(),1); }
	float mutualInformation(const float* A,const float* B, int size,int stride)		// in bits
	{
		clearHistograms();
		addToHistograms(A,B, size,stride);
		float enta,entb,entab;
		entropies(enta,entb,entab);				// in bits
		float mi = (enta+entb-entab);			// KL divergence of joint distribution and product of independent distributions
		return (m_normalized) ? mi*rcp(entab) : mi;
	}

	void	printHistograms() const				// DEBUG function
	{
		printf("Histogram_A:\n");
		for(int i=0;i<N;i++)
			printf("%d\t", (int)m_histogramA[i]);
		printf("\n");
		printf("Histogram_B:\n");
		for(int i=0;i<N;i++)
			printf("%d\t", (int)m_histogramB[i]);
		printf("\n");
		printf("Histogram_AB (A_on_columns):\n");
		for(int i=0;i<N;i++)
		{
			for(int j=0;j<N;j++)
				printf("%d\t", (int)m_histogramAB[i*N+j]);
			printf("\n");
		}
	}

private:
	inline int		quantize(float v) const		// maps 2 stddevs (98% of normalized input) to valid range, clamps outside it
	{
		v = (v+2)/4;							// interesting range [-2,2] -> [0,1]
		v*= N-1;								// interesting range [0,N-1]
		int bucket = (int)(v+0.5f);				// Q: best rounding mode?
		return clamp(bucket,0,N-1);
	}	

	inline float	myLog2(float v) const		{ return log(v)/log(2.f); }

	void clearHistograms()
	{
		for(int i=0;i<N;i++)
		{
			m_histogramA[i] = 0;
			m_histogramB[i] = 0;
		}
		for(int i=0;i<N*N;i++)
		{
			m_histogramAB[i] = 0;
		}
		m_numSamples = 0;
	}

	inline void addToHistograms(const float* A,const float* B, int size,int stride)	{ for(int i=0,j=0;j<size;i+=stride,j++) addToHistograms(A[i],B[i]); }
	inline void addToHistograms(float A,float B)
	{
		int a = quantize(A);
		int b = quantize(B);
		m_histogramA[a]++;
		m_histogramB[b]++;
		m_histogramAB[a*N+b]++;
		m_numSamples++;
	}

	void entropies(float& enta,float& entb,float& entab) const	// in bits
	{
		enta  = 0.f;
		entb  = 0.f;
		for(int i=0;i<N;i++)
		{
			if(m_histogramA[i])
			{
				float pa = m_histogramA[i]/m_numSamples;
				enta += -pa*myLog2(pa);
			}
			if(m_histogramB[i])
			{
				float pb = m_histogramB[i]/m_numSamples;
				entb += -pb*myLog2(pb);
			}
		}
		entab = 0.f;
		for(int i=0;i<N*N;i++)
		{
			if(m_histogramAB[i])
			{
				float pab = m_histogramAB[i]/m_numSamples;
				entab += -pab*myLog2(pab);
			}
		}
	}

	int			 m_numSamples;		// # samples in histograms (used for computing probability)
	Array<float> m_histogramA;
	Array<float> m_histogramB;
	Array<float> m_histogramAB;
	bool		 m_normalized;
};

//-----------------------------------------------------------------------
// remove mean and divide by stddev
//-----------------------------------------------------------------------

inline void normalize(float* A, int size,int stride)
{
	float E  = 0.f;
	float E2 = 0.f;
	for(int i=0,j=0;j<size;i+=stride,j++)
	{
		E  += A[i];
		E2 += A[i]*A[i];
	}
	E  /= size;
	E2 /= size;

	const float mean   = E;
	const float stddev = sqrt(max(0.f,E2 - E*E));	// max() avoids accidental NaN
	const float ooStddev = rcp(stddev);				// deals with stddev=0
	for(int i=0,j=0;j<size;i+=stride,j++)
	{
		A[i] = (A[i]-mean)*ooStddev;
	}
}
inline void normalize(Array<float>& A)	{ normalize(A.getPtr(), A.getSize(),1); }
inline void normalize(SOAAccessor& A)	{ normalize(A.getPtr(), A.getSize(),A.getStride()); }

//-----------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------

class Random2D
{
public:
		  Random2D		(U32 seed=0)		{ reset(seed); }
	void  reset			(U32 seed)			{ m_random.reset(seed); }

	Vec2f getGaussian	(float stddev, Vec2f mean=0.f)
	{
		// Box-Muller method
		float S,V1,V2;
		do{
			V1 = 2*m_random.getF32()-1;		// [-1,1[
			V2 = 2*m_random.getF32()-1;		// [-1,1[
			S = V1*V1 + V2*V2;
		} while(S>=1);

		Vec2f v;
		v.x = sqrt(-2 * log(S) / S) * V1;	// mean 0, variance 1
		v.y = sqrt(-2 * log(S) / S) * V2;	// mean 0, variance 1
		return mean + stddev * v;			// adjust mean and std dev
	}

	Vec2f	getCircle	(float R)			// [-R,R]
	{
		Vec2f v;
		do{
			v.x = 2*m_random.getF32()-1;	// [-1,1[
			v.y = 2*m_random.getF32()-1;	// [-1,1[
		} while(v.length()>=1);
		return v*R;
	}

	U32		getU32		(void)			{ return m_random.getU32(); }
	F32		getF32		(void)			{ return m_random.getF32(); }

private:
	Random m_random;
};

}; // FW