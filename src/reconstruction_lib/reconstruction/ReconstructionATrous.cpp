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
Implementation of A-Trous [Dammertz 2010]. 
Also supports true cross bilateral filtering.
*/

#pragma warning(disable:4127)		// conditional expression is constant
#include "ReconstructionATrous.hpp"

namespace FW
{

//-----------------------------------------------------------------------
// Sets scene-dependent parameters
//-----------------------------------------------------------------------

#define MONKEYS2
//#define SAN_MIGUEL

#ifdef MONKEYS2
	#define PRE_MULTIPLY_ALBEDO		// outgoing radiance (default is incident)
									// more definitions later
#endif

//-----------------------------------------------------------------------
// Options
//-----------------------------------------------------------------------

const bool MULTI_CORE				= true;
const bool USE_ATROUS				= true;				// i.e. insert 2^it-1 zeros? If set to false, the filter degenerates to cross bilateral.
const bool USE_ATROUS_JITTER		= true;				// Remove banding artifacts from A-Trous. This is an unpublished trick, which helps a lot.

const bool STOP_COLOR				= false;			// Dammertz used this, but it doesn't really work with path traced input.
const bool STOP_NORMAL				= true;				// Additional stopping conditions. The best combination is scene-dependent.
const bool STOP_POSITION			= true;
const bool STOP_NORMAL2				= true;
const bool STOP_POSITION2			= false;

const float MAX_COLOR_STDDEV		= 4.f;				// Otherwise high-energy spikes prevent all filtering. Dammertz confirmed they used 4.0.

const int FILTER_WIDTH[] = {5,5,5,5,5,0};				// N passes of A-Trous, zero terminates
//const int FILTER_WIDTH[] = {23,0};  					// Cross bilateral filter (corresponds to approximately 3 rounds of A-Trous)

//-----------------------------------------------------------------------
// Entry point (may be changed/merged to other methods eventually)
//-----------------------------------------------------------------------

void Reconstruction::reconstructATrous(const UVTSampleBuffer& sbuf, Image& resultImage, Image* debugImage, float aoLength)
{
	profileStart();

	printf("\n");
	printf("A-Trous Filtering\n");
	ATrous atrous(&resultImage,debugImage, sbuf, aoLength);

	profileEnd();
}

//-----------------------------------------------------------------------
// Init
//-----------------------------------------------------------------------

ATrous::ATrous(Image* resultImage, Image* debugImage, const UVTSampleBuffer& sbuf, float aoLength)
{
	if(sbuf.isIrregular())
		fail("ATrous implementation does not support irregular sample buffers");
	if(sbuf.getVersion()<2.f)
		fail("ATrous works only with sample buffer >= V2.0");

	const int CID_PRI_NORMAL   = sbuf.getChannelID(CID_PRI_NORMAL_SMOOTH_NAME  );
	const int CID_SEC_ORIGIN   = sbuf.getChannelID(CID_SEC_ORIGIN_NAME  );
	const int CID_SEC_NORMAL   = sbuf.getChannelID(CID_SEC_NORMAL_NAME  );
	const int CID_SEC_HITPOINT = sbuf.getChannelID(CID_SEC_HITPOINT_NAME);
	const int CID_ALBEDO       = sbuf.getChannelID(CID_ALBEDO_NAME);

	const int w = sbuf.getWidth();
	const int h = sbuf.getHeight();

	m_image      = resultImage;
	m_debugImage = debugImage;
	m_w			 = w;
	m_h			 = h;

	// fetch samples to local structs (removes invalid entries)

	m_numSamples .reset(w*h);
	m_firstSample.reset(w*h);

	for(int i=0;i<w*h;i++)
	{
		m_numSamples [i] = 0; 
		m_firstSample[i] = -1;
	}

	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	for(int i=0;i<sbuf.getNumSamples(x,y);i++)
	{
		SampleVector s;
		s.xy = sbuf.getSampleXY(x,y,i);
		s.n  = sbuf.getSampleExtra<Vec3f>(CID_PRI_NORMAL,   x,y,i);	// primary hit origin
		s.p  = sbuf.getSampleExtra<Vec3f>(CID_SEC_ORIGIN,   x,y,i);	// primary hit normal
		s.n2 = sbuf.getSampleExtra<Vec3f>(CID_SEC_NORMAL,   x,y,i);	// secondary hit origin
		s.p2 = sbuf.getSampleExtra<Vec3f>(CID_SEC_HITPOINT, x,y,i);	// secondary hit normal
		s.a  = sbuf.getSampleExtra<Vec3f>(CID_ALBEDO,       x,y,i);

	#ifdef PRE_MULTIPLY_ALBEDO
		s.c  = sbuf.getSampleColor(x,y,i).getXYZ() * s.a;			// NOTE: multiplied by albedo!!
	#else
		s.c  = sbuf.getSampleColor(x,y,i).getXYZ();
	#endif

		if(aoLength>0)
			s.c  = (s.p2-s.p).length()<=aoLength ? 0.f : 1.f;

		if(s.p.max() > 1e10f)										// invalid sample (this is due to technical reasons inside PBRT)
			continue;

		const int numSamples = m_samples.getSize();
		const int pixelIndex = y*w+x;
		if(m_numSamples[pixelIndex]==0)
			m_firstSample[pixelIndex] = numSamples;
		m_numSamples[pixelIndex]++;

		m_samples.add( s );
		m_inputColors .add( s.c );
		m_outputColors.add( s.c );
	}

	// compute initial stddevs

	SampleVector E;
	SampleVector E2;
	SampleVector mx;
	for(int i=0;i<m_samples.getSize();i++)
	{
		const SampleVector& s = m_samples[i];
		for(int k=0;k<SampleVector::getSize();k++)
		{
			E [k] += s[k];
			E2[k] += s[k]*s[k];
			mx[k]  = max(mx[k],s[k]);
		}
	}
	E. divide( m_samples.getSize() );
	E2.divide( m_samples.getSize() );

	SampleVector stddev;
	for(int k=0;k<SampleVector::getSize();k++)
		stddev[k] = max(0.f, sqrt(E2[k] - E[k]*E[k]));

	// set scene-dependent fudge factors... (obtained via manual search)

	float p_scale  = 1;
	float n_scale  = 1;
	float n2_scale = 1;
	float p2_scale = 1;

#ifdef MONKEYS2
	p_scale  = 1/3.5f;
	n_scale  = 1;
	n2_scale = 1.5f;
	p2_scale = 1;
#endif

#ifdef SAN_MIGUEL
	p_scale  = 1/5.f;
	n_scale  = 1/2.5f;
	n2_scale = 2.f;
	p2_scale = 1;
#endif

	m_stddev.p  = stddev.p.length()		* p_scale;
	m_stddev.n  = stddev.n.length()		* n_scale;
	m_stddev.p2 = stddev.p2.length()	* p2_scale;
	m_stddev.n2 = stddev.n2.length()	* n2_scale;
	m_stddev.a  = dot(Vec3f(0.30f,0.59f,0.11f),stddev.a);
	m_stddev.c  = dot(Vec3f(0.30f,0.59f,0.11f),stddev.c);

	printf("Intial statistics:\n");
	printf("stddev:\n");
	printf("Position  = %f\n", m_stddev.p[0]);
	printf("Normal    = %f\n", m_stddev.n[0]);
	printf("Position2 = %f\n", m_stddev.p2[0]);
	printf("Normal2   = %f\n", m_stddev.n2[0]);
	printf("Color     = %f\n", m_stddev.c[0]);
	printf("max:\n");
	printf("Color     = %f\n", mx.c.length());

	m_stddev.c = min(MAX_COLOR_STDDEV,mx.c.length());	// Dammertz: "we choose initial stddev_color to include variations of the scale of the maximum intensity"

	// how much energy did we have initially?

	double initialEnergy = 0;
	for(int i=0;i<m_inputColors.getSize();i++)
		initialEnergy += dot(m_inputColors[i], Vec3f(0.30f,0.59f,0.11f));

	// multiple filter iterations

	for(int iteration=0;FILTER_WIDTH[iteration];iteration++)
	{
		filter(iteration);								// writes output colors, also writes image (useful for debug purposes)

		for(int i=0;i<m_inputColors.getSize();i++)		// copy output -> input
			m_inputColors[i]  = m_outputColors[i];
	}

	// what happened to overall energy?

	double finalEnergy = 0;
	for(int i=0;i<m_inputColors.getSize();i++)
		finalEnergy += dot(m_inputColors[i], Vec3f(0.30f,0.59f,0.11f));

	if(finalEnergy > initialEnergy)
		printf("  WARNING: processing added %.2f%% of energy\n", 100*(finalEnergy/initialEnergy-1));
	if(finalEnergy < initialEnergy)
		printf("  WARNING: processing lost %.2f%% of energy\n", 100*(1-finalEnergy/initialEnergy));
	
	// Generate output image.

	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	{
		Vec4f pixelColor(0);
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			const int index = getSampleIndex(x,y,i);
	#ifdef PRE_MULTIPLY_ALBEDO
			pixelColor += Vec4f( getOutputColor(index), 1 );
	#else
			pixelColor += Vec4f( getOutputColor(index)*getSampleVector(index).a, 1 );
	#endif
		}
		if(!pixelColor.w)
			pixelColor = Vec4f(0,0,0,1);	// background color

		pixelColor *= rcp(pixelColor.w);
		resultImage->setVec4f(Vec2i(x,y),pixelColor);
	}
}

//-----------------------------------------------------------------------
// Filter iteration
//-----------------------------------------------------------------------

void ATrous::filter(int iteration)
{
	const String name1 = String("Filtering (iteration ") + String(iteration) + String(", ") + String(FILTER_WIDTH[iteration]) + String("x") + String(FILTER_WIDTH[iteration]);
	const String name2 = String("Filtering (iteration ") + String(iteration) + String(")");
	profilePush( name2.getPtr() );

	const int h = m_h;

	Array<ATrousTask> tasks;
	tasks.reset(h);
	MulticoreLauncher launcher;
	for(int y=0;y<h;y++)
	{
		tasks[y].init(this, iteration);
		if(MULTI_CORE)	launcher.push(ATrousTask::filter, &tasks[y], y, 1);	// multi core
		else			tasks[y].filter(y);									// single core
	}
	launcher.popAll( name1.getPtr() );

	profilePop();
}

//-----------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------

void ATrous::ATrousTask::filter(int y)
{
	Random random(y);

	const int w = m_scope->m_w;
	const int h = m_scope->m_h;
	const int fw = FILTER_WIDTH[m_iteration];	// filter width
	const int fr = fw/2;						// filter radius
	const int atrous_scale    = (USE_ATROUS) ? 1<<m_iteration : 1;
	const int atrous_numzeros = atrous_scale - 1;

	for(int x=0;x<w;x++)
	{
		const Vec2i pixel(x,y);

		// DEBUG features: focus on one pixel or compute only a part of the image

	#define DEBUG_PIXEL  Vec2i(-1,-1)			// (-1,-1) = Disabled
		if(DEBUG_PIXEL!=Vec2i(-1,-1) && DEBUG_PIXEL!=pixel)
			continue;

	//#define PARTIAL_IMAGE
	#ifdef PARTIAL_IMAGE
		const int xmin = 100;
		const int ymin = 0;
		const int xmax = 200;
		const int ymax = h;
		if(x<xmin || x>=xmax || y<ymin || y>=ymax)
			continue;
	#endif

		// Filter all samples in this pixel

		for(int i=0;i<m_scope->getNumSamples(x,y);i++)
		{
			const int indexi = m_scope->getSampleIndex(x,y,i);
			const SampleVector& si = m_scope->getSampleVector(indexi);
			const Vec3f& ci = m_scope->getInputColor(indexi);
			Vec4f color(0);

			for(int dx=-fr;dx<=fr;dx++)
			for(int dy=-fr;dy<=fr;dy++)
			{
				int sx = x+dx*atrous_scale;						// effectively inserts atrous_scale-1 zeros in between
				int sy = y+dy*atrous_scale;

				if(USE_ATROUS_JITTER && atrous_numzeros>0 && !(dx==0 && dy==0))
				{
					sx += S32(random.getU32()%(atrous_numzeros)) - atrous_numzeros/2;		// jitter [-atrous_numzeros/2,atrous_numzeros/2]
					sy += S32(random.getU32()%(atrous_numzeros)) - atrous_numzeros/2;		// jitter [-atrous_numzeros/2,atrous_numzeros/2]
				}

				if(sx<0 || sy<0 || sx>=w || sy>=h)
					continue;

				for(int j=0;j<m_scope->getNumSamples(sx,sy);j++)
				{
					const int indexj = m_scope->getSampleIndex(sx,sy,j);
					const SampleVector& sj = m_scope->getSampleVector(indexj);
					const Vec3f& cj = m_scope->getInputColor(indexj);

					// spatial (screen)
					const float xydist2  = (sj.xy - si.xy).lenSqr();
					const float xyradius = fw/2.f*atrous_scale;
					const float xystddev = xyradius / 2.f;									// 2 stddevs (98%) at the filter border
					float d = xydist2/(2*sqr(xystddev));

					// color
					if(STOP_COLOR)
					{
						const float cdist2  = (cj - ci).lenSqr();
						const float cstddev = m_scope->m_stddev.c[0] / (1<<m_iteration);	// make more strict every iteration (Dammerz text)
						d += cdist2/(2*sqr(cstddev));
					}

					// normal
					if(STOP_NORMAL)
					{
						const float ndist2  = (sj.n - si.n).lenSqr();
						const float nstddev = m_scope->m_stddev.n[0] / (1<<m_iteration);	// make more strict every iteration (Dammertz pseudocode)
						d += ndist2/(2*sqr(nstddev));
					}

					// position (world)
					if(STOP_POSITION)
					{
						const float pdist2  = (sj.p - si.p).lenSqr();
						const float pstddev = m_scope->m_stddev.p[0] / (1<<m_iteration);	// make more strict every iteration (Dammertz email)
						d += pdist2/(2*sqr(pstddev));
					}

					// secondary normal
					if(STOP_NORMAL2)
					{
						const float ndist2  = (sj.n2 - si.n2).lenSqr();
						const float nstddev = m_scope->m_stddev.n2[0] / (1<<m_iteration);
						d += ndist2/(2*sqr(nstddev));
					}

					// secondary position (world)
					if(STOP_POSITION2)
					{
						const float pdist2  = (sj.p2 - si.p2).lenSqr();
						const float pstddev = m_scope->m_stddev.p2[0] / (1<<m_iteration);
						d += pdist2/(2*sqr(pstddev));
					}
					// combined
					const float weight = exp( -d );
					color += weight*Vec4f(cj,1);
				}
			}

			color *= rcp(color.w);
			((ATrous*)m_scope)->setOutputColor(indexi, color.getXYZ());
		}
	}
}

} // 