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