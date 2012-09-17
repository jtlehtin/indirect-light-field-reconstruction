/*
Implementation of random parameter filtering [Sen et al. 2012].

NOTES:
- We share "alpha" for all color channels. Sounds like Sen's independent treatment of color channels might discolor the image.
- Sen's original HDR clamp results in discoloration because it's done separately for each color channel. The tech rep also lacks a fabs(). Also, 1 stddev seems very aggressive.
- We could do clamping in color filtering to force outliers to be filtered. Defocus doesn't work well without. Overall HDR should be solved in a proper way.
- The scale of mutual information is [0,joint entropy]. Dubious to just add them up. if NORMALIZE_MI defined, we divide by joint entropy to get [0,1]. This has only a very minor effect, however.
- Filtering depends on #color features. I.e. GGG and G give different results because distances are simply added up. This is not entirely satisfactory.
- W_r_c depends on the number of random parameters and position features. If we use 4 random parameters instead of 2, is it OK that W_r_c is affected?
- RGB features fail to notice dependency on dark and uniformly colored areas. Could consider using W as an additional color channel.
- Techrep probably has an error in Algorithm 3 L13 (alpha). We believe the factor of 2 shouldn't be there.


Per-sample info:

SCREEN POSITION:
- xy (2)

RANDOM PARAMETERS:
- t  (1)
- uv (2)
- first reflection direction (3)

FEATURES:
- primary hit point		(3)
- primary normal		(3)
- secondary hit point	(3)
- secondary normal		(3)
- albedo/texture value	(3)		// Sen actually stores ALL texture lookups

SAMPLE COLOR:
- rgb (3)

TOTAL:
- 2+6+15+3 = 26

- Sen and Darabi say: "store up to 27 floats per sample".
*/


#pragma warning(disable:4127)		// conditional expression is constant
#include "ReconstructionRPF.hpp"


namespace FW
{
//#define MONKEYS2
#define SAN_MIGUEL


#ifdef SAN_MIGUEL
	//#define PRE_MULTIPLY_ALBEDO							// DISABLED: filter incident light
	const bool MULTI_CORE				= true;

	const float Jouni					= 0.02f;			// Sen: 0.02 for path tracing, 0.002 for others
	const bool ENABLE_CLUSTERING		= true;				// Sen: true
	const bool ENABLE_TECHREP_ALPHABETA	= true;				// Sen: true

	const bool STOP_COLOR				= true;				// Sen: true (but one really shouldn't do this for path traced images)
	const bool ENABLE_HDR_CLAMP			= STOP_COLOR;		// This is actually very important. It does a lot more than just "HDR", basically it's an "outlier" removal tool.
	const bool ENABLE_HDR_CLAMP_EP		= STOP_COLOR;		// Hack: put most of the lost energy back

	const int  MI_NUM_BUCKETS			= 5;				// Sen: 5 should be pretty close to Sen's implementation, based on the cited Matlab code
	const bool MI_NORMALIZATION			= false;			// Sen: false. the effect is surprisingly small

	// Iterative process, 0 terminates
	const int   BOX_WIDTH[] = {55,35,17,7,0};				// As suggested by Sen
	//const float FSAMPLING[] = {0.02f,0.04f,0.3f,0.5f,0};	// For fast prototyping
	const float FSAMPLING[] = {0.5f,0.50f,0.50f,0.50f,0};	// Sen uses always 50% of samples but that's SLOW

	// DEBUG
//	const int   BOX_WIDTH[] = {7,0};
//	const float FSAMPLING[] = {0.5f,0};
#endif


#ifdef MONKEYS2
	#define PRE_MULTIPLY_ALBEDO								// filter outgoing light

	const bool MULTI_CORE				= true;

	const float Jouni					= 0.01f;			// Sen: 0.02 for path tracing, 0.002 for others
	const bool ENABLE_CLUSTERING		= true;				// Sen: true
	const bool ENABLE_TECHREP_ALPHABETA	= true;				// Sen: true

	const bool STOP_COLOR				= true;				// Sen: true (but one really shouldn't do this for path traced images)
	const bool ENABLE_HDR_CLAMP			= STOP_COLOR;		// This is actually very important. It does a lot more than just "HDR", basically it's an "outlier" removal tool.
	const bool ENABLE_HDR_CLAMP_EP		= STOP_COLOR;		// Hack: put most of the lost energy back

	const int  MI_NUM_BUCKETS			= 5;				// Sen: 5 should be pretty close to Sen's implementation, based on the cited Matlab code
	const bool MI_NORMALIZATION			= false;			// Sen: false. the effect is surprisingly small

	// Iterative process, 0 terminates
	const int   BOX_WIDTH[] = {55,35,17,7,0};				// Sen
	const float FSAMPLING[] = {0.02f,0.04f,0.2f,0.5f,0};	// For fast prototyping
#endif

//-----------------------------------------------------------------------
// Single-pixel debugging
//-----------------------------------------------------------------------

//const Vec2i DEBUG_PIXEL 			= Vec2i(638,38);	// out of focus background

#ifndef DEBUG_PIXEL
const Vec2i DEBUG_PIXEL 			= Vec2i(-1,-1);		// disabled
#endif

//-----------------------------------------------------------------------
// Entry point (may be changed/merged to other methods eventually)
//-----------------------------------------------------------------------

void Reconstruction::reconstructRPF(const UVTSampleBuffer& sbuf, Image& resultImage, Image* debugImage, float aoLength)
{
	profileStart();

	// Print useful info

	printf("\n");
	printf("Random Parameter Filtering\n");

	printf("\n");
	printf("%d ScreenPositions\n", ScreenPosition::getSize());
	printf("%d RandomParams\n", RandomParams::getSize());
	printf("%d SceneFeatures\n", SceneFeatures::getSize());
	printf("%d ColorFeatures\n", ColorFeatures::getSize());
	printf("%d total SampleVector entries\n", SampleVector::getSize());
	printf("\n");

	// Init shared variables and filter

	RPF rpf(&resultImage,debugImage, sbuf, aoLength);

	profileEnd();
}

//-----------------------------------------------------------------------
// Init
//-----------------------------------------------------------------------

RPF::RPF(Image* resultImage, Image* debugImage, const UVTSampleBuffer& sbuf, float aoLength)
{
	if(sbuf.isIrregular())
		fail("RPF implementation does not support irregular sample buffers");

	CID_PRI_MV       = sbuf.getChannelID(CID_PRI_MV_NAME      );
	CID_PRI_NORMAL   = sbuf.getChannelID(CID_PRI_NORMAL_SMOOTH_NAME  );
	CID_ALBEDO       = sbuf.getChannelID(CID_ALBEDO_NAME      );
	CID_SEC_ORIGIN   = sbuf.getChannelID(CID_SEC_ORIGIN_NAME  );
	CID_SEC_HITPOINT = sbuf.getChannelID(CID_SEC_HITPOINT_NAME);
	CID_SEC_MV       = sbuf.getChannelID(CID_SEC_MV_NAME      );
	CID_SEC_NORMAL   = sbuf.getChannelID(CID_SEC_NORMAL_NAME  );
	CID_DIRECT       = sbuf.getChannelID(CID_DIRECT_NAME      );
	CID_SEC_ALBEDO   = sbuf.getChannelID(CID_SEC_ALBEDO_NAME  );
	CID_SEC_DIRECT   = sbuf.getChannelID(CID_SEC_DIRECT_NAME  );

	const int w = sbuf.getWidth();
	const int h = sbuf.getHeight();
	const int s = sbuf.getNumSamples(); 

	m_image      = resultImage;
	m_debugImage = debugImage;
	m_w			 = w;
	m_h			 = h;
	m_spp		 = s;

	// fetch samples to local structs (replaces invalid input samples with average of pixel's valid samples)

	Random random(242);
	int	statsValidSamples[4] = {0};

	m_unNormalizedSamples.reset(w*h*s);
	m_inputColors. reset(w*h*s);
	m_outputColors.reset(w*h*s);

	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	{
		Array<int> validSampleIndices;
		SampleVector avg(0);

		for(int i=0;i<s;i++)
		{
			const int index = getSampleIndex(x,y,i);
			SampleVector& s = m_unNormalizedSamples[index];
			s.fetch(sbuf,x,y,i);								// some samples in sbuf may be invalid

	#ifdef PRE_MULTIPLY_ALBEDO
			s.c.rgb *= sbuf.getSampleExtra<Vec3f>(CID_ALBEDO,x,y,i);
	#endif

			if(aoLength>0)
				s.c.rgb  = (s.f.p2-s.f.p1).length()<=aoLength ? 0.f : 1.f;

			if(s.f.isValid())
			{
				validSampleIndices.add(index);
				avg += s;
			}
		}

		// average of valid samples
		const int numValidSamples = validSampleIndices.getSize();
		if(numValidSamples>0)
			avg.divide(s);

		if(numValidSamples<4)
			statsValidSamples[numValidSamples]++;

		// "fix" invalid samples
		for(int i=0;i<s;i++)
		{
			const int index = getSampleIndex(x,y,i);
			SampleVector& s = m_unNormalizedSamples[index];
			if(!s.f.isValid())
			{
				if(numValidSamples>0)
				{
					// Option 1: copy one of the valid samples, but make radiance black
					const int cidx = random.getU32() % numValidSamples;
					SampleVector& s2 = m_unNormalizedSamples[ validSampleIndices[cidx] ];
					s = s2;
					s.c.rgb = Vec3f(0,0,0);

					// Option 2: copy average, but make radiance black
	//				s = avg;		// fill in average
	//				s.c.rgb = Vec3f(0,0,0);
				}
				else
				{
					s = avg;
				}
			}
			m_inputColors [index] = s.c.rgb;
			m_outputColors[index] = s.c.rgb;
		}
	}

	for(int i=0;i<4;i++)
	{
		if(statsValidSamples[i])
			printf("WARNING: %d pixels had only %d valid samples\n", statsValidSamples[i], i);
	}

	// how much energy did we have initially?

	double initialEnergy = 0;
	for(int i=0;i<m_inputColors.getSize();i++)
		initialEnergy += dot(m_inputColors[i], Vec3f(0.30f,0.59f,0.11f));

	// multiple iterations

	for(int iteration=0;BOX_WIDTH[iteration];iteration++)
	{
		filter(iteration);				// writes output colors, also writes image (useful for debug purposes at least)

		for(int i=0;i<w*h*s;i++)		// copy output -> input
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
		for(int i=0;i<s;i++)
		{
			const int index = getSampleIndex(x,y,i);
	#ifdef PRE_MULTIPLY_ALBEDO
			pixelColor += Vec4f(getInputColor(index),1);
	#else
			pixelColor += Vec4f(getInputColor(index)*getUnNormalizedSampleVector(index).f.albedo,1);
	#endif		
		}
		if(!pixelColor.w)
			pixelColor = Vec4f(0,0,0,1);	// background
		pixelColor *= rcp(pixelColor.w);
		resultImage->setVec4f(Vec2i(x,y),pixelColor);
	}
}

void RPF::filter(int iteration)
{
	const String name1 = String("Filtering (iteration ") + String(iteration) + String(", ") + String(BOX_WIDTH[iteration]) + String("x") + String(BOX_WIDTH[iteration]) + String(" @ ") + String(int(FSAMPLING[iteration]*100)) + String("%)");
	const String name2 = String("Filtering (iteration ") + String(iteration) + String(")");
	profilePush( name2.getPtr() );

	const int h = m_h;

	Array<RPFTask> tasks;
	tasks.reset(h);
	MulticoreLauncher launcher;
	for(int y=0;y<h;y++)
	{
		tasks[y].init(this, iteration);
		if(MULTI_CORE)	launcher.push(RPFTask::filter, &tasks[y], y, 1);	// multi core
		else			tasks[y].filter(y);									// single core
	}
	launcher.popAll( name1.getPtr() );

	profilePop();
}

//-----------------------------------------------------------------------
// Clustering
//-----------------------------------------------------------------------

void RPFTask::getPixelMeanAndStddev(const Vec2i& pixel,SceneFeatures& E, SceneFeatures& stddev) const
{
	const int x = pixel.x;
	const int y = pixel.y;
	const int s = m_scope->m_spp;
	SceneFeatures E2;

	for(int f=0;f<SceneFeatures::getSize();f++)
	{
		E [f] = 0.f;
		E2[f] = 0.f;
	}

	for(int i=0;i<s;i++)
	{
		const SceneFeatures& v = m_scope->getUnNormalizedSampleVector(x,y,i).f;
		for(int f=0;f<SceneFeatures::getSize();f++)
		{
			E [f] += v[f];
			E2[f] += v[f]*v[f];
		}
	}

	for(int f=0;f<SceneFeatures::getSize();f++)
	{
		E [f] /= s;
		E2[f] /= s;
		stddev[f] = sqrt(max(0.f,E2[f] - E[f]*E[f]));	// max() avoids accidental NaNs
	}
}

void RPFTask::determineNeighborhood(Array<int>& neighborIndices, const Vec2i& pixel,int boxWidth,int M, Random2D& random) const
{
	Array<int> histogram;	// DEBUG
	Set<int> set;			// Currently not used

	const int w = m_scope->m_w;
	const int h = m_scope->m_h;
	const int s = m_scope->m_spp;

	if(DEBUG_PIXEL!=Vec2i(-1,-1))
	{
		histogram.reset(boxWidth*boxWidth);
		for(int i=0;i<boxWidth*boxWidth;i++)
			histogram[i] = 0;
	}

	// mean and standard deviation of features inside this pixel
	SceneFeatures pixelMean;
	SceneFeatures pixelStddev;
	getPixelMeanAndStddev(pixel, pixelMean, pixelStddev);

	// start with all the samples from the pixel itself (color filtering actually relies on this)
	neighborIndices.clear();
	for(int i=0;i<s;i++)
	{
		const int index = m_scope->getSampleIndex(pixel.x,pixel.y,i);
		neighborIndices.add( index );
		//set.add( index );
	}

	for(int i=0;i<M-s;i++)
	{
		// Sen: "select a random sample from inside the box but outside the pixel with distribution based on stddev_p"
		// - we select the pixel in this way, and then randomly select one samples from the pixel
		int x=pixel.x;
		int y=pixel.y;
		do
		{
			const float stddev_p = boxWidth/4.f;
			Vec2f offset = random.getGaussian(stddev_p);		// zero mean, radius = width/2 = 2*stddev
			if(offset.abs().max() >= boxWidth/2.f)				// inside the box?
				continue;
			x = pixel.x + int(floor(offset.x+0.5f));			// pixel center (0.5,0.5)
			y = pixel.y + int(floor(offset.y+0.5f));
		}
		while(Vec2i(x,y)==pixel ||								// same pixel -> retry
			  x<0 || y<0 || x>=w || y>=h);						// outside the screen -> retry

		// select a random sample inside the chosen pixel
		const int index = m_scope->getSampleIndex(x,y,random.getU32()%s);

		// compare features
		const SceneFeatures& v = m_scope->getUnNormalizedSampleVector(index).f;
		bool flag = true;

		if(ENABLE_CLUSTERING)
		for(int f=0;f<SceneFeatures::getSize() && flag;f++)
		{
			const float lim = (f<FIRST_NON_POSITION_FEATURES) ? 30.f : 3.f;
			if( abs(v[f]-pixelMean[f]) > lim*pixelStddev[f] &&
			   (abs(v[f]-pixelMean[f]) > 0.1f || pixelStddev[f] > 0.1f) )
				flag = false;
		}

		// similar enough -> accept
		if(flag && !set.contains(index))
		{
			neighborIndices.add( index );
			//set.add( index );											// force each sample to be included only once (doesn't make much of a difference)
			if(DEBUG_PIXEL!=Vec2i(-1,-1))
				histogram[(y-pixel.y+boxWidth/2)*boxWidth + (x-pixel.x+boxWidth/2)]++;
		}
	}

	if(DEBUG_PIXEL!=Vec2i(-1,-1))
	{
		for(int y=0;y<boxWidth;y++,printf("\n"))
		for(int x=0;x<boxWidth;x++)
			printf("%d,", histogram[y*boxWidth+x]);
		printf("\n");
	}
}

//-----------------------------------------------------------------------
// Weights
//-----------------------------------------------------------------------

Vec3f RPFTask::computeWeights(ColorFeatures& alpha,SceneFeatures& beta,float& W_r_c, const Array<SampleVector>& normalizedSamples, int t)
{
	MutualInformation<MI_NUM_BUCKETS> mi(MI_NORMALIZATION);

	const int p_offset = offsetof(SampleVector,p) / sizeof(float);
	const int r_offset = offsetof(SampleVector,r) / sizeof(float);
	const int f_offset = offsetof(SampleVector,f) / sizeof(float);
	const int c_offset = offsetof(SampleVector,c) / sizeof(float);

	// Compute dependency for colors 

	m_D_rk_c.set(0);								// kth random parameter vs. ALL color channels
	m_D_pk_c.set(0);								// kth screen position vs. ALL color channels
	m_D_fk_c.set(0);								// kth scene feature vs. ALL color channels

	for(int l=0;l<ColorFeatures::getSize();l++)
	{
		const SOAAccessor cl(normalizedSamples,c_offset+l);
		for(int k=0;k<RandomParams::getSize();  k++)	m_D_rk_c[k] += mi.mutualInformation(cl, SOAAccessor(normalizedSamples,r_offset+k));
		for(int k=0;k<ScreenPosition::getSize();k++)	m_D_pk_c[k] += mi.mutualInformation(cl, SOAAccessor(normalizedSamples,p_offset+k));
		for(int k=0;k<SceneFeatures::getSize(); k++)	m_D_fk_c[k] += mi.mutualInformation(cl, SOAAccessor(normalizedSamples,f_offset+k));
	}

	// Compute dependency for scene features

	m_D_fk_rl.reset(SceneFeatures::getSize());		// kth scene feature vs. lth random parameter
	m_D_fk_pl.reset(SceneFeatures::getSize());		// kth scene feature vs. lth screen position
	m_D_fk_cl.reset(SceneFeatures::getSize());		// kth scene feature vs. lth color feature

	for(int k=0;k<SceneFeatures::getSize();k++)
	{
		const SOAAccessor fk(normalizedSamples,f_offset+k);
		for(int l=0;l<RandomParams::getSize();  l++)	m_D_fk_rl[k][l] = mi.mutualInformation(fk, SOAAccessor(normalizedSamples,r_offset+l));	// Eq. 2
		for(int l=0;l<ScreenPosition::getSize();l++)	m_D_fk_pl[k][l] = mi.mutualInformation(fk, SOAAccessor(normalizedSamples,p_offset+l));	// Eq. 3
		for(int l=0;l<ColorFeatures::getSize(); l++)	m_D_fk_cl[k][l] = m_D_fk_c[k]/ColorFeatures::getSize();		// NOTE: average of color channels
	}

	// Compute aggregates

	const float D_r_c = m_D_rk_c.sum();				// ALL color channels vs. ALL random parameters	(Eq. 4)
	const float D_p_c = m_D_pk_c.sum();				// ALL color channels vs. ALL screen positions	(Eq. 5)
	const float D_f_c = m_D_fk_c.sum();				// ALL color channels vs. ALL scene features	(Eq. 6)
	const float D_a_c = D_r_c+D_p_c+D_f_c;			// ALL color channels vs. ALL features

	W_r_c = D_r_c * rcp(D_r_c + D_p_c);				// Position vs. Random: How much do random parameters tell about the color? [0,1] (Eq. 11)

	// NOTE: we set the same alpha to all color channels
	if(ENABLE_TECHREP_ALPHABETA)
		alpha.set( max(1-(1+0.1f*t)*W_r_c,0.f) );	// Error in Tech rep:	max(1-2*(1+0.1f*t)*W_r_c,0.f)
	else
		alpha.set( 1 - W_r_c );						// Paper

	if(DEBUG_PIXEL!=Vec2i(-1,-1))
	{
		printf("D_r_c = %.1f%% (ck: ", 100.f*D_r_c*rcp(D_a_c));	for(int k=0;k<RandomParams::getSize();  k++)	printf("%.1f%%, ", 100.f*m_D_rk_c[k]*rcp(D_a_c));	printf(")\n");
		printf("D_p_c = %.1f%% (pk: ", 100.f*D_p_c*rcp(D_a_c));	for(int k=0;k<ScreenPosition::getSize();k++)	printf("%.1f%%, ", 100.f*m_D_pk_c[k]*rcp(D_a_c));	printf(")\n");
		printf("D_f_c = %.1f%% (fk: ", 100.f*D_f_c*rcp(D_a_c));	for(int k=0;k<SceneFeatures::getSize(); k++)	printf("%.1f%%, ", 100.f*m_D_fk_c[k]*rcp(D_a_c));	printf(")\n");
		printf("W_r_c = %.5f [0,1]\n", W_r_c);
		printf("alpha = %.5f [0,1]\n", alpha[0]);
	}

	// Compute dependency for scene features

	for(int k=0;k<SceneFeatures::getSize();k++)
	{
		const float D_fk_r = m_D_fk_rl[k].sum();	// kth scene feature vs. ALL random parameters
		const float D_fk_p = m_D_fk_pl[k].sum();	// kth scene feature vs. ALL screen positions
		const float D_fk_c = m_D_fk_cl[k].sum();	// kth scene feature vs. ALL color features

		float W_fk_r = D_fk_r * rcp(D_fk_r + D_fk_p);	// Eq. 9	Position vs. Random: How much do random parameters tell about kth feature? [0,1]
		float W_fk_c = D_fk_c * rcp(D_a_c);				// Eq. 12	How much of the overall information about color comes from kth feature? [0,1]

		if(ENABLE_TECHREP_ALPHABETA)
			beta[k] = W_fk_c * max(1-(1+0.1f*t)*W_fk_r,0.f);	// Tech rep.
		else
			beta[k] = W_fk_c * (1 - W_fk_r);					// Paper

		if(DEBUG_PIXEL!=Vec2i(-1,-1))
		{
			printf("SceneFeature %d\n", k);
			printf("D_fk_r = %.1f%% (rl: ", 100.f*D_fk_r*rcp(D_a_c));	for(int l=0;l<RandomParams::getSize();  l++)	printf("%.1f%%, ", 100.f*m_D_fk_rl[k][l]*rcp(D_a_c));	printf(")\n");
			printf("D_fk_p = %.1f%%\n", 100.f*D_fk_p*rcp(D_a_c));
			printf("D_fk_c = %.1f%%\n", 100.f*D_fk_c*rcp(D_a_c));
			printf("W_fk_r = %.5f [0,1]\n", W_fk_r);
			printf("W_fk_c = %.5f [0,1]\n", W_fk_c);
			printf("beta   = %.5f [0,1]\n", beta[k]);
		}
	}

	return Vec3f(m_D_pk_c[0]+m_D_pk_c[1], m_D_rk_c[0]+m_D_rk_c[1], m_D_rk_c[2]);		// DEBUG: effect of xy, uv, t on rgb
/**/

/*
	//---------------------------------------------------------------------------------------------
	// FOLLOWING SEN'S TECH REP PSEUDOCODE (kept for now if we need to debug, otherwise the above is nicer)
	//---------------------------------------------------------------------------------------------

	// Compute dependency for colors 

	float D_r_c = 0.f;			// ALL color channels vs. ALL random parameters
	float D_p_c = 0.f;			// ALL color channels vs. ALL screen positions
	float D_f_c = 0.f;			// ALL color channels vs. ALL scene features
		  W_r_c = 0.f;

	SceneFeatures D_c_fk_tmp;	// ALL color channnels vs. kth scene feature
	for(int l=0;l<SceneFeatures::getSize();l++)	
		D_c_fk_tmp[l] = 0.f;

	for(int k=0;k<ColorFeatures::getSize();k++)
	{
		const SOAAccessor ck(normalizedSamples,c_offset+k);

		float D_r_ck = 0.f;	// kth color channel vs. ALL random parameters
		for(int l=0;l<RandomParams::getSize();l++)	
			D_r_ck += mi.mutualInformation(ck, SOAAccessor(normalizedSamples,r_offset+l));	// Eq. 4
		
		float D_p_ck = 0.f;	// kth color channel vs. ALL screen positions
		for(int l=0;l<ScreenPosition::getSize();l++)	
			D_p_ck += mi.mutualInformation(ck, SOAAccessor(normalizedSamples,p_offset+l));	// Eq. 5

		float D_f_ck = 0.f;	// kth color channel vs. ALL features
		for(int l=0;l<SceneFeatures::getSize();l++)	
		{
			float D_fl_ck = mi.mutualInformation(ck, SOAAccessor(normalizedSamples,f_offset+l));
			D_f_ck += D_fl_ck;														// Eq. 6
			D_c_fk_tmp[l] += D_fl_ck;												// Eq. 7	Note: k<->l discrepancy (also in Sen et al.)
		}

		const float W_r_ck = D_r_ck / (D_r_ck + D_p_ck + eps);						// Eq. 10	Position vs. Ranom: How much do random parameters tell about the color? [0,1]

//		alpha[k] = max(1 - 2*(1+0.1f*t)*W_r_ck, 0.f);								// Tech rep. TODO: Surely this is wrong (esp. 2*)
		alpha[k] = 1 - W_r_ck;

		W_r_c += W_r_ck;															// Eq. 11
		D_r_c += D_r_ck;	// ALL color channels vs. ALL random parameters			// Eq. 8
		D_p_c += D_p_ck;	// ALL color channels vs. ALL screen positions
		D_f_c += D_f_ck;	// ALL color channels vs. ALL scene features
	}

	W_r_c /= ColorFeatures::getSize();
	const float D_all_c = D_r_c + D_p_c + D_f_c;

	if(DEBUG_PIXEL!=Vec2i(-1,-1))
	{
		printf("D_r_c = %.5f\n", D_r_c);
		printf("D_p_c = %.5f\n", D_p_c);
		printf("D_f_c = %.5f\n", D_f_c);
		printf("W_r_c = %.5f [0,1]\n", W_r_c);
		printf("alpha = %.5f [0,1] color channel 0\n", alpha[0]);
	}

	// Compute dependency for scene features

	for(int k=0;k<SceneFeatures::getSize();k++)
	{
		const SOAAccessor fk(normalizedSamples,f_offset+k);

		float D_r_fk = 0.f;	// kth scene feature vs. ALL random parameters
		for(int l=0;l<RandomParams::getSize();l++)	
			D_r_fk += mi.mutualInformation(fk, SOAAccessor(normalizedSamples,r_offset+l));	// Eq. 2

		float D_p_fk = 0.f;	// kth scene feature vs. ALL screen positions
		for(int l=0;l<ScreenPosition::getSize();l++)
			D_p_fk += mi.mutualInformation(fk, SOAAccessor(normalizedSamples,p_offset+l));	// Eq. 3

							// kth scene feature vs. ALL screen positions
		float D_c_fk = D_c_fk_tmp[k];												// Eq. 7 (computed above)

		float W_r_fk = D_r_fk / (D_r_fk + D_p_fk + eps);							// Eq. 9	Position vs. Random: How much do random parameters tell about kth feature? [0,1]
		float W_c_fk = D_c_fk / D_all_c;											// Eq. 12	How much of the overall information about color comes from kth feature? [0,1]

//		beta[k] = W_c_fk * max(1 - (1+0.1f*t)*W_r_fk,0.f);							// Tech rep.
		beta[k] = W_c_fk * (1 - W_r_fk);

		if(DEBUG_PIXEL!=Vec2i(-1,-1))
		{
			printf("SceneFeature %d\n", k);
			printf("D_r_fk = %.5f\n", D_r_fk);
			printf("D_p_fk = %.5f\n", D_p_fk);
			printf("D_c_fk = %.5f\n", D_c_fk);
			printf("W_r_fk = %.5f [0,1]\n", W_r_fk);
			printf("W_c_fk = %.5f [0,1]\n", W_c_fk);
			printf("beta   = %.5f [0,1]\n", beta[k]);
		}
	}
	return Vec3f(W_r_c,W_r_c,W_r_c);	// HACK: u+v+t
/**/
}

//-----------------------------------------------------------------------
// DEBUG: compute and print the full mutual information matrix
//-----------------------------------------------------------------------

inline int getCategory(int i)
{
	const int p_offset = offsetof(SampleVector,p) / sizeof(float);
	const int r_offset = offsetof(SampleVector,r) / sizeof(float);
	const int f_offset = offsetof(SampleVector,f) / sizeof(float);
	const int c_offset = offsetof(SampleVector,c) / sizeof(float);

	if(i>=p_offset && i<r_offset)	return 0;	// P
	if(i>=r_offset && i<f_offset)	return 1;	// R
	if(i>=f_offset && i<c_offset)	return 2;	// F
	if(i>=c_offset)					return 3;	// C
	fail(0);
	return -1;
}

inline const char* getName(int i)	// NOTE: hardcoded, max 4 chars
{
	switch(i)
	{
	case 0:  return "x";
	case 1:  return "y";
//	case 2:  return "u";			// NEW_RANDOM_PARAM
//	case 3:  return "v";
//	case 4:  return "t";
	case 2:  return "dirx";
	case 3:  return "diry";
	case 4:  return "dirz";
	case 5:  return "p1.x";
	case 6:  return "p1.y";
	case 7:  return "p1.z";
	case 8:  return "p2.x";
	case 9:  return "p2.y";
	case 10: return "p2.z";
	case 11: return "n1.x";
	case 12: return "n1.y";
	case 13: return "n1.z";
	case 14: return "n2.x";
	case 15: return "n2.y";
	case 16: return "n2.z";
	case 17: return "a.r";
	case 18: return "a.g";
	case 19: return "a.b";
	case 20: return "r";
	case 21: return "g";
	case 22: return "b";
	default: fail(0); return "INV";
	}
}

inline const char* getCategoryName(int i)	// NOTE: hardcoded
{
	switch(i)
	{
	case 0:  return "Pos";
	case 1:  return "Rand";
	case 2:  return "Feat";
	case 3:  return "Col";
	default: fail(0); return "INV";
	}
}

void RPFTask::printAllWeights(const Array<SampleVector>& samples) const
{
	MutualInformation<MI_NUM_BUCKETS> mi(MI_NORMALIZATION);
	const int NUM_ELEMENTS   = SampleVector::getSize();
	const int NUM_CATEGORIES = 4;

	struct Entry
	{
		int	i,j;
		float key;
	};

	// evaluate all element pairs
	Array<Entry> ElementPairs;
	for(int i=0;i<NUM_ELEMENTS;i++)
	for(int j=0;j<NUM_ELEMENTS;j++)
	{
		const SOAAccessor ci(samples,i);
		const SOAAccessor cj(samples,j);
		Entry e;
		e.i = i;
		e.j = j;
		e.key = mi.mutualInformation(ci,cj);
		ElementPairs.add(e);
	}

	// category - category
	Array<Entry> CategoryPairs;
	for(int i=0;i<NUM_CATEGORIES;i++)
	for(int j=0;j<NUM_CATEGORIES;j++)
	{
		float sum = 0;
		for(int k=0;k<ElementPairs.getSize();k++)
		{
			const Entry& e = ElementPairs[k];
			if(getCategory(e.i)==i && getCategory(e.j)==j)
				sum += e.key;
		}
		Entry e;
		e.i = i;
		e.j = j;
		e.key = sum;
		CategoryPairs.add(e);
	}

	// element - category
	Array<Entry> ElementCategoryPairs;
	for(int i=0;i<NUM_CATEGORIES;i++)
	for(int j=0;j<NUM_ELEMENTS;j++)
	{
		float sum = 0;
		for(int k=0;k<ElementPairs.getSize();k++)
		{
			const Entry& e = ElementPairs[k];
			if(getCategory(e.i)==i && e.j==j)
				sum += e.key;
		}
		Entry e;
		e.i = i;
		e.j = j;
		e.key = sum;
		ElementCategoryPairs.add(e);
	}

	printf("-----------------------------------------\n");
	printf("Element-Element mutual information:\n");
	printf("      ");
	for(int i=0;i<NUM_ELEMENTS;i++)
			printf(" %-4s, ", getName(i));
	printf("\n");
	for(int i=0;i<NUM_ELEMENTS;i++, printf("\n"))
	for(int j=0;j<NUM_ELEMENTS;j++)
	{
		const Entry& e = ElementPairs[i*NUM_ELEMENTS+j];
		if(j==0)								printf("%-4s: ", getName(i));
		if(getCategory(e.i)<getCategory(e.j))	printf("%-.3f, ", e.key);
		else									printf(" --- , ");
	}
	printf("\n");
	printf("-----------------------------------------\n");
	printf("Element-Category mutual information:\n");
	printf("      ");
	for(int i=0;i<NUM_ELEMENTS;i++)
			printf(" %-4s, ", getName(i));
	printf("\n");
	for(int i=0;i<NUM_CATEGORIES;i++, printf("\n"))
	for(int j=0;j<NUM_ELEMENTS;j++)
	{
		const Entry& e = ElementCategoryPairs[i*NUM_ELEMENTS+j];
		if(j==0)								printf("%-4s: ", getCategoryName(i));
		if(e.i<getCategory(e.j))				printf("%-.3f, ", e.key);
		else									printf(" --- , ");
	}
	printf("\n");
	printf("-----------------------------------------\n");
	printf("Category-Category mutual information:\n");
	printf("      ");
	for(int i=0;i<NUM_CATEGORIES;i++)
			printf(" %-4s, ", getCategoryName(i));
	printf("\n");
	for(int i=0;i<NUM_CATEGORIES;i++, printf("\n"))
	for(int j=0;j<NUM_CATEGORIES;j++)
	{
		const Entry& e = CategoryPairs[i*NUM_CATEGORIES+j];
		if(j==0)								printf("%-4s: ", getCategoryName(i));
		if(e.i<e.j)								printf("%-.3f, ", e.key);
		else									printf(" --- , ");
	}
	printf("\n");
	printf("-----------------------------------------\n");

}

//-----------------------------------------------------------------------
// Cross bilateral filter
// NOTE: assumes [0,spp-1] are the samples from pixel itself
//-----------------------------------------------------------------------

Vec4f RPFTask::filterColorSamples(const ColorFeatures& alpha,const SceneFeatures& beta,float W_r_c, const Array<SampleVector>& normalizedSamples,const Array<int>& neighborIndices)
{
	Vec3f* outputColors = ((RPF*)m_scope)->getOutputColorPtr(neighborIndices[0]);	// NOTE: [0] must be the first sample of this pixel

	const int s = m_scope->m_spp;

	const float var_8 = Jouni;								// Sen: for path traced scenes 0.002, others 0.02
	const float var   = 8*var_8/s;							// Q: Spatial filter may need to get larger but why range filter?

//	const float var_c = var / sqr(1-W_r_c);					// From Sen's techrep, this line can cause div-by-zero. Numerically stable version below.
//	const float var_f = var_c;
//	const float scale_c = -1/(2*var_c);
//	const float scale_f = -1/(2*var_f);

	const float scale_c = -sqr(1-W_r_c) / (2*var);			// -1/(2*var_c) = -1/(2*var/sqr(1-W_r_c)) = -sqr(1-W_r_c) / (2*var);
	const float scale_f = -sqr(1-W_r_c) / (2*var);

	// filter

	for(int i=0;i<s;i++)									// for each sample in this pixel
	{
		Vec4f sampleColor(0);
		for(int j=0;j<normalizedSamples.getSize();j++)		// for each sample in the neighborhood
		{
			const int index = neighborIndices[j];
			Vec4f color(m_scope->getInputColor(index),1);

			// Sen: spatial Gaussian dropped because samples drawn proportional to that
			// Eq. 18
			float dist_c = 0;
			if(STOP_COLOR)
				for(int k=0;k<ColorFeatures::getSize();k++)
					dist_c += alpha[k] * sqr(normalizedSamples[i].c[k] - normalizedSamples[j].c[k]);

			float dist_f = 0;
			for(int k=0;k<SceneFeatures::getSize();k++)
				dist_f += beta[k] * sqr(normalizedSamples[i].f[k] - normalizedSamples[j].f[k]);

			const float w_ij = exp(scale_c*dist_c + scale_f*dist_f);
			sampleColor += w_ij * color;
		}
		sampleColor *= rcp(sampleColor.w);
		outputColors[i] = sampleColor.getXYZ();
	}

	// handle HDR issues (replaces outliers with pixel mean) NOTE: Sen's tech rep does this independently for each color channel, which is clearly wrong.
	// we also reduce the energy loss by putting the lost energy back.

	if(ENABLE_HDR_CLAMP)
	{
		Vec3f E(0);
		Vec3f E2(0);
		for(int i=0;i<s;i++)
		{
			E  += outputColors[i];
			E2 += sqr(outputColors[i]);
		}
		E  /= float(s);
		E2 /= float(s);
		const Vec3f mean = E;
		const Vec3f var  = max(Vec3f(0), E2 - E*E);
		const Vec3f stddev(	sqrt(var[0]),sqrt(var[1]),sqrt(var[2]) );

		Vec3f meanAfter(0);
		for(int i=0;i<s;i++)
		{
			if( fabs(outputColors[i][0]-mean[0]) > 1*stddev[0] ||	// R	TODO: 2*stddev?
				fabs(outputColors[i][1]-mean[1]) > 1*stddev[1] ||	// G
				fabs(outputColors[i][2]-mean[2]) > 1*stddev[2] )	// B
				outputColors[i] = mean;
			meanAfter += outputColors[i];
		}
		meanAfter /= float(s);

		if(ENABLE_HDR_CLAMP_EP)
		{
			Vec3f lostEnergyPerSample = (mean-meanAfter);
			for(int i=0;i<s;i++)
				outputColors[i] += lostEnergyPerSample;
		}
	}

	// return pixel color (useful for debugging)

	Vec4f pixelColor(0);
	for(int i=0;i<s;i++)
		pixelColor += Vec4f(outputColors[i],1);
	pixelColor *= rcp(pixelColor.w);
	return pixelColor;
}

//-----------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------

//#define PARTIAL_IMAGE
#ifdef PARTIAL_IMAGE
		const int ymin = 200;
		const int ymax = 300;
#else
		const int ymin = -1;
		const int ymax = 1000000;
#endif

void RPFTask::filter(int y)
{
	if(!(y>=ymin && y<=ymax))
		return;

	Random2D				random2D(y);
	Array<int>				neighborIndices;	// indices to scope->m_unNormalizedSamples
	Array<SampleVector>		normalizedSamples;	// normalized samples in the neighborhood

	const int w = m_scope->m_w;
	const int s = m_scope->m_spp;
	Image& image = *(m_scope->m_image);

	for(int x=0;x<w;x++)
	{
		const Vec2i pixel(x,y);

		if(DEBUG_PIXEL!=Vec2i(-1,-1) && DEBUG_PIXEL!=pixel)
			continue;

		// Determine neighborhood (collects indices of the samples that seem to belong to the same cluster)

		const int b = BOX_WIDTH[m_iteration];							// box width for this iteration
		const int M = int(b*b*s*FSAMPLING[m_iteration]);				// how many samples to consider?
		determineNeighborhood(neighborIndices, pixel,b,M, random2D);

		// Normalize the neighborhood, attribute by attribute

		normalizedSamples.clear();
		for(int i=0;i<neighborIndices.getSize();i++)
			normalizedSamples.add( m_scope->getUnNormalizedSampleVector(neighborIndices[i]) );

		for(int i=0;i<SampleVector::getSize();i++)
		{
			SOAAccessor attribute(normalizedSamples,i);
			normalize( attribute );
		}

		// Compute weights

		ColorFeatures alpha;
		SceneFeatures beta;
		float W_r_c;
		Vec3f W = computeWeights(alpha,beta,W_r_c, normalizedSamples, m_iteration);

		if(DEBUG_PIXEL != Vec2i(-1,-1))
			printAllWeights(normalizedSamples);

		// DEBUG: Tweak alpha,beta:
		// - 0,0 => blur
		// - 1,0 => preserve color differences (doesn't remove MC noise)
		// - 0,1 => cross bilateral filter based on scene features
		// - 1,1 => preserves all differences, all noise remains

//		for(int k=0;k<ColorFeatures::getSize();k++)	{ alpha[k] = 0.f; W_r_c = 1; }
//		for(int k=0;k<SceneFeatures::getSize();k++)	{ beta[k]  = 0.f; }

		// Filter samples

//		Vec4f color(W_r_c,1);
		Vec4f color = filterColorSamples(alpha,beta,W_r_c, normalizedSamples,neighborIndices);
		image.setVec4f(pixel, color);
	}
}

} // 