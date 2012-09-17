/*
Implementation of A-Trous [Dammertz 2010]
*/
#include "Reconstruction.hpp"

namespace FW
{

#define ATROUS_ACCESSOR(X)	static int	 getSize()		{ return sizeof(X)/sizeof(float); }\
				const float& operator[](int i) const	{ return ((float*)this)[i]; } \
				float& operator[](int i)				{ return ((float*)this)[i]; } \
				void   operator+=(const X& s)			{ for(int i=0;i<getSize();i++) (*this)[i] += s[i]; } \
				void   divide(int s)					{ for(int i=0;i<getSize();i++) (*this)[i] /= float(s); } \
				X(float f=0.f)							{ set(f); } \
				void  set(float f)						{ for(int i=0;i<getSize();i++) (*this)[i] = f; } \
				float sum() const						{ float s=0; for(int i=0;i<getSize();i++) s+=(*this)[i]; return s; } \

//-----------------------------------------------------------------------
// Core functionality, data shared among all tasks.
//-----------------------------------------------------------------------

class ATrous
{
	// Immutable sample data. Color is modified every iteration and thus double buffered.
	struct SampleVector
	{
		Vec2f	xy;
		Vec3f	n;
		Vec3f	p;
		Vec3f	n2;
		Vec3f	p2;
		Vec3f	a;
		Vec3f	c;			// INITIAL color
		ATROUS_ACCESSOR(SampleVector);
	};

public:
			ATrous				(Image* resultImage, Image* debugImage, const UVTSampleBuffer& sbuf, float aoLength=0.f);

	void	filter				(int iteration);

	const SampleVector&	getSampleVector		(int index) const					{ return m_samples[index]; }

	const Vec3f&		getInputColor		(int index) const					{ return m_inputColors[index]; }
	const Vec3f&		getOutputColor		(int index) const					{ return m_outputColors[index]; }
	void				setOutputColor		(int index, const Vec3f& v)			{ m_outputColors[index]=v; }

private:
	// Multi-core task
	class ATrousTask
	{
	public:
		void	init	(const ATrous* scope,int t)			{ m_scope = scope; m_iteration=t; }

		static	void	filter	(MulticoreLauncher::Task& task) { ATrousTask* fttask = (ATrousTask*)task.data; fttask->filter(task.idx); }
				void	filter	(int y);
	private:
		const ATrous*	m_scope;		// container for shared data
		int				m_iteration;	// which iteration?
	};


	int					getNumSamples	(int x,int y) const						{ return m_numSamples [y*m_w+x]; }
	int					getSampleIndex	(int x,int y,int i) const				{ return m_firstSample[y*m_w+x]+i; }

	Image*					m_image;
	Image*					m_debugImage;

	int						m_w;
	int						m_h;
	Array<int>				m_numSamples;					// number of VALID samples in this pixel
	Array<int>				m_firstSample;					// index of the sample in this pixel

	Array<SampleVector>		m_samples;						// "raw" sample data
	Array<Vec3f>			m_inputColors;					// current iteration uses these colors
	Array<Vec3f>			m_outputColors;					// ... and writes these colors

	SampleVector			m_stddev;
};

//-----------------------------------------------------------------------

}; // FW