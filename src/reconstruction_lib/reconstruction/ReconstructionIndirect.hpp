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

#pragma once
#include "Reconstruction.hpp"


namespace FW
{
//#define ENABLE_BACKFACE_CULLING			// PBRT does not use backface culling. Disabled by default.

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

static const float UVPLANE_DISTANCE = 1.f;	// In light field parameterization. This value shouldn't affect the results, kept for debug purposes.


class ReconstructIndirect
{
public:
	ReconstructIndirect		(const UVTSampleBuffer& sbuf, int numReconstructionRays, String rayDumpFileName=String(""), float aoLength=0, bool print=true, bool enableCUDA=false, bool enableMotion=false, Vec4i rectangle=Vec4i(0));
	void	filterImage		(Image& image, Image* debugImage);
	
	void	filterImageCuda	(Image& image);
	void	shrinkCuda		(void);

private:

	enum
	{
		NBITS				= 21,				// for morton code, per dimension
		ROOT				= 0,				// must not be changed!
		MAX_LEAF_SIZE		= 32,
		SUBSAMPLE_SBUF		= 1,
		K1					= 12,
		K2					= 12,
		ANISOTROPIC_SCALE	= 2,
		NUM_DENSITY_TASKS	= 32,
		NUM_SAMPLE_COUNTERS = 10,
	};

	enum BloatMode
	{
		SPHERE,
		CIRCLE,
		POINT
	};

	struct SortEntry
	{
		U64		code;			// morton
		Vec3i	idx;			// index in sample buffer
	};

	struct Node
	{
		Node()							{ child0=-1; child1=-1; s0=-1; s1=-1; bbmin=FW_F32_MAX; bbmax=-FW_F32_MAX; bbminT1=FW_F32_MAX; bbmaxT1=-FW_F32_MAX; }
		bool	isLeaf() const			{ return child0==-1; }
		Node(const Node& n0,const Node& n1)
		{
			child0 = -1;
			child1 = -1;
			s0     = -1;
			s1     = -1;
			ns     = n0.ns + n1.ns;
			bbmin  = min(n0.bbmin,n1.bbmin);
			bbmax  = max(n0.bbmax,n1.bbmax);
			bbminT1= min(n0.bbminT1,n1.bbminT1);
			bbmaxT1= max(n0.bbmaxT1,n1.bbmaxT1);
		};

		inline float	getExpectedCost(float t) const							{ return ns *  getSurfaceArea(t); }
		inline float	getSurfaceArea(float t) const							{ const Vec3f bb = getBBMax(t)-getBBMin(t); return 2*(bb.x*bb.y + bb.y*bb.z + bb.z*bb.x); }	// box
		inline Vec3f	getCenter(float t) const								{ return (getBBMin(t)+getBBMax(t))/2.f; }
		inline float	getRadius(float t) const								{ return (getBBMax(t)-getBBMin(t)).length()/2.f; }
		inline float	getDistance(const Vec3f& p,float t) const				{ return max(getBBMin(t)-p, p-getBBMax(t), Vec3f(0)).length(); }							// Thanks, Eberly
		inline bool		inside(const Vec3f& p,float t) const					{ return (p.x>=getBBMin(t).x && p.x<=getBBMax(t).x) && (p.y>=getBBMin(t).y && p.y<=getBBMax(t).y) && (p.z>=getBBMin(t).z && p.z<=getBBMax(t).z); }

		inline float	getFarthestCornerDist(const Vec4f& pleq,float t) const	{ return dot(pleq, Vec4f((pleq.x>=0 ? getBBMax(t).x : getBBMin(t).x), (pleq.y>=0 ? getBBMax(t).y : getBBMin(t).y), (pleq.z>=0 ? getBBMax(t).z : getBBMin(t).z), 1)); }
		inline float	getNearestCornerDist (const Vec4f& pleq,float t) const	{ return dot(pleq, Vec4f((pleq.x<=0 ? getBBMax(t).x : getBBMin(t).x), (pleq.y<=0 ? getBBMax(t).y : getBBMin(t).y), (pleq.z<=0 ? getBBMax(t).z : getBBMin(t).z), 1)); }

		inline bool intersect(const Vec3f& idir,const Vec3f& ood,float t) const	{ float tenter; return intersect(tenter,idir,ood,t); }
		inline bool intersect(float& tenter, const Vec3f& idir,const Vec3f& ood,float time) const
		{
			const Vec3f ta = getBBMin(time) * idir - ood;	// inaccurate -- the node needs to have been bloated a little bit
			const Vec3f tb = getBBMax(time) * idir - ood;
			const Vec3f t0 = min(ta,tb);				// minimum of per-axis enter times
			const Vec3f t1 = max(ta,tb);				// maximum of per-axis enter times

						tenter = max(t0.max(), 0.f);
			const float texit  = t1.min();

            return (tenter <= texit);
		}

		int		child0,child1;
		int		s0,s1;			// sample indices (inc,exc)
		int		ns;				// total number of sample under this node

		Vec3f	bbmin,bbmax;	// @ t=0 (not renamed yet because of dependencies)
		Vec3f	bbminT1,bbmaxT1;// @ t=1

		inline Vec3f		getBBMin(float t) const							{ return (s_motionEnabled) ? bbmin + t*(bbminT1-bbmin) : bbmin; }
		inline Vec3f		getBBMax(float t) const							{ return (s_motionEnabled) ? bbmax + t*(bbmaxT1-bbmax) : bbmax; }
		static bool s_motionEnabled;
	};

	Node*		buildRecursive		(const Array<SortEntry>& codes, int& sampleIdx, const int MAX_LEAF, const U64 octreeCode,const int octreeBitPos);
	float		getHierarchyArea	(int nodeIdx, bool leafOnly=false) const;
	void		validateNodeBounds	(int nodeIdx, BloatMode mode);

	struct ReconSample;

	class DensityTask
	{
	public:
		void	init(ReconstructIndirect* scope) { m_scope = scope; }

		static	void	compute			(MulticoreLauncher::Task& task) { DensityTask* ttask = (DensityTask*)task.data; ttask->compute(task.idx); }
				void	compute			(int taskIdx);

		static	void	shrink			(MulticoreLauncher::Task& task) { DensityTask* ttask = (DensityTask*)task.data; ttask->shrink(task.idx); }
				void	shrink			(int taskIdx);

		ReconstructIndirect*	m_scope;

		U64		m_numSteps;
		U64		m_numLeaf;
		U64		m_numIter;
		U64		m_numShrank;
		U64		m_numSelfHits;
		double	m_averageVMF;
	};

public:
	struct Sample;

	const Array<Sample>& getSamples() const { return m_samples; }

	class FilterTask
	{
	public:
		void	init(ReconstructIndirect* scope)									{ m_scope = scope; m_image = NULL; m_debugImage=NULL; }
		void	init(ReconstructIndirect* scope, Image* image, Image* debugImage)	{ m_scope = scope; m_image = image; m_debugImage=debugImage; }

		Vec4f	sampleRadiance(const Vec3f& o,const Vec3f& d,float t=0.f);	// w is approx 1.0 if found support, w=0 otherwise

		void				clearNumUniqueInputSamplesUsed()						{ m_supportSet.clear(); }
		int					getNumUniqueInputSamplesUsed() const					{ return m_supportSet.getSize(); }
		const Set<int>&		getSupportSet() const									{ return m_supportSet; }
		const Sample&		getSample( int i ) const								{ return m_scope->m_samples[ i ]; }

		static inline float	vMFfromBandwidth( float bw )							{ return 4.0f*sqrt( max(0.f,bw) ); }

		// Can s be used for reconstructing radiance to direction d towards p?
		// - tp is point on the tangent plane of the splat
		inline float		interpolationWeight( const Vec3f& p, const Vec3f& d, const Vec3f& tp, const Sample&s )
		{
			// The core question here is how does the angular validity fall off on the surface of a splat.

			// The splat sent radiance to this direction
			const Vec3f splatDir = -(s.sec_hitpoint-s.sec_origin).normalized();

			// Angle between query direction and the splat's original direction
			float anglecos = dot( splatDir, -d );

			// Support of von Mises-Fischer to the query direction
			float bw = vMFfromBandwidth( m_scope->m_sbuf->getSampleW( s.origIndex.x, s.origIndex.y, s.origIndex.z ) );
			float vMF = expf( bw * anglecos - bw );	// vMF but normalized to [0,1]

			// Scale the vMF support based on a near field tweak. How large is the splat compared to the length of the ray?
			float distWeight = min(1.f, s.radius/(tp-p).length());
			const float spatialWeight = max( 0.0f, 1.0f - (tp-s.sec_hitpoint).length()/s.radius );
			return vMF * min(1.f, spatialWeight/distWeight);
		}

		static	void	filter	(MulticoreLauncher::Task& task) { FilterTask* fttask = (FilterTask*)task.data; fttask->filter(task.idx); }
				void	filter	(int y);

		static	void	filterPBRT	(MulticoreLauncher::Task& task) { FilterTask* fttask = (FilterTask*)task.data; fttask->filterPBRT(task.idx); }
				void	filterPBRT	(int y);

		static	void	filterDofMotion	(MulticoreLauncher::Task& task) { FilterTask* fttask = (FilterTask*)task.data; fttask->filterDofMotion(task.idx); }
				void	filterDofMotion	(int y);

		struct Stats
		{
			Stats() { memset(this,0,sizeof(Stats)); }
			void	operator+=(const Stats& src)	{ Vec2d* d=(Vec2d*)this; const Vec2d* s=(const Vec2d*)(&src); for(int i=0;i<sizeof(Stats)/sizeof(Vec2d);i++) d[i] += s[i]; }
			void	newOutput() const				{ Vec2d* d=(Vec2d*)this; for(int i=0;i<sizeof(Stats)/sizeof(Vec2d);i++) d[i] += Vec2d(0,1); }
			Vec2d	numTraversalSteps;
			Vec2d	numSamplesTested;
			Vec2d	numSamplesAccepted;
			Vec2d	numSamplesFirstSurface;
			Vec2d	numSurfaces;
			Vec2d	numMissingSupport;
			Vec2d	numSamplesFirstSurfaceTable[NUM_SAMPLE_COUNTERS+1];
			Vec2d	vMFSupport;
		};
		Stats					m_stats;

	private:
		struct LocalParameterization
		{
			LocalParameterization(const Vec3f& o,const Vec3f& d,float t)
			{
				time = t;
				orig = o;
				dir	 = d;

				// orient dual planes
				stplane = Vec4f(dir, -dot(dir,orig));							// oriented according to the direction of the ray
				uvplane = stplane - Vec4f(0,0,0,UVPLANE_DISTANCE);				// at a specified distance from stplane

				// set up transformation between coordinate systems
				const Mat3f planeToCamera = orthogonalBasis(dir);
				const Mat3f cameraToPlane = planeToCamera.inverted();

				cameraToSTPlane.setCol(0, Vec4f( cameraToPlane.getCol(0),0) );	// copy rotation (TODO: so inconvenient...)
				cameraToSTPlane.setCol(1, Vec4f( cameraToPlane.getCol(1),0) );
				cameraToSTPlane.setCol(2, Vec4f( cameraToPlane.getCol(2),0) );
				cameraToSTPlane.setCol(3, Vec4f(-cameraToPlane * orig,1) );

				cameraToUVPlane.setCol(0, Vec4f( cameraToPlane.getCol(0),0) );
				cameraToUVPlane.setCol(1, Vec4f( cameraToPlane.getCol(1),0) );
				cameraToUVPlane.setCol(2, Vec4f( cameraToPlane.getCol(2),0) );
				cameraToUVPlane.setCol(3, Vec4f(-cameraToPlane * (orig + UVPLANE_DISTANCE*dir),1) );

				STPlaneToCamera = cameraToSTPlane.inverted();
				UVPlaneToCamera = cameraToUVPlane.inverted();

				// prepare ray for fast traversal
				const float ooeps = 1e-20f;
				idir.x = 1.0f / (fabs(d.x)>ooeps ? dir.x : (dir.x<0 ? -ooeps : ooeps));
				idir.y = 1.0f / (fabs(d.y)>ooeps ? dir.y : (dir.y<0 ? -ooeps : ooeps));
				idir.z = 1.0f / (fabs(d.z)>ooeps ? dir.z : (dir.z<0 ? -ooeps : ooeps));
				ood = orig * idir;
			}

			Vec3f	orig;				// shoot secondary from here
			Vec3f	dir;				// direction of the secondary ray
			Vec4f	stplane;			// oriented according to the direction of this ray
			Vec4f	uvplane;			// 
			Mat4f	cameraToSTPlane;		
			Mat4f	cameraToUVPlane;		
			Mat4f	STPlaneToCamera;		
			Mat4f	UVPlaneToCamera;		
			Vec3f	idir;				// for fast traversal
			Vec3f	ood;
			float	time;
		};

		bool isOriginInsideConvexHull(int lo,int hi, const LocalParameterization& lp) const
		{
			float Rscale;
			switch(hi-lo)
			{
			case 1:		Rscale = 0.5f; break;
			case 2:		Rscale = 0.6f; break;
			case 3:		Rscale = 0.7f; break;
			case 4:		Rscale = 1.f;  break;
			case 5:		Rscale = 1.f;  break;
			case 6:		Rscale = 1.f;  break;
			default:	Rscale = 1.f;  break;
			}

			if(Rscale==1.f)
				return true;

			InsideConvexHull h;
			const Array<ReconSample>& samples = m_reconSamples;
			const Mat3f cameraToQuery = orthogonalBasis(lp.dir).transposed();	// (symmetric matrix)

			for(int i=lo;i<hi;i++)
			{
				const ReconSample& r = samples[i];
				const Sample& s = m_scope->m_samples[ r.index ];

				const Vec3f pip = intersectRayPlane(lp.orig,(s.getHitPoint(lp.time)-lp.orig), lp.uvplane);	// intersection point in camera space
				const Vec2f xy  = (lp.cameraToUVPlane * pip).getXY();										// on the UV plane
				const float R   = UVPLANE_DISTANCE * s.radius / r.zdist;									// project the (isotropic) size to splane

				const Vec3f N         = cameraToQuery * s.sec_normal;			// In query's coordinate system (z aligned with the query ray)
				const float cosAngle  = fabs(dot(N,Vec3f(0,0,1)));				// cos of the angle between N and query

				Vec2f N2d = N.getXY().normalized();								// .xy of the normal is the minor axis
				if(N2d==Vec2f(0))												// -- except when it's degenerate
					N2d = Vec2f(1,0);

				const float minorScale = Rscale * R * cosAngle;
				const float majorScale = Rscale * R;
				const Vec2f minorAxis  = minorScale * Vec2f( N2d.x, N2d.y);
				const Vec2f majorAxis  = majorScale * Vec2f( N2d.y,-N2d.x);
				h.add( xy + minorAxis );
				h.add( xy - minorAxis );
				h.add( xy + majorAxis );
				h.add( xy - majorAxis );

				const bool useDiagonal = true;
				if(!useDiagonal)
					continue;

				const float diagScale  = Rscale * R * sqrt(1*1 + cosAngle*cosAngle);
				const Vec2f diag1Axis  = diagScale  * Vec2f( (N2d.x+N2d.y)/2, (N2d.y-N2d.x)/2 );
				const Vec2f diag2Axis  = diagScale  * Vec2f( (N2d.x-N2d.y)/2, (N2d.y+N2d.x)/2 );

				h.add( xy + diag1Axis );
				h.add( xy - diag1Axis );
				h.add( xy + diag2Axis );
				h.add( xy - diag2Axis );
			}

			return h.originInside();
		}

		void	collectSamples	(const LocalParameterization& lp);
		Vec2i	getNextSurface	(Vec2i samplesInPrevSurface, const LocalParameterization& lp);

		static inline int	ternaryCompare	(float a, float b, float eps)	{ return (fabs(a-b)<eps) ? 0 : (a<b ? -1 : 1); }	// 0==don't care
		static inline bool	ternaryEqual	(int a, int b)					{ return a==0 || b==0 || a==b; }

		ReconstructIndirect*	m_scope;
		Image*					m_image;
		Image*					m_debugImage;
		Vec2i					m_pixelIndex;	// debug feature
		Set<int>				m_supportSet;

		float					m_vMFSupport;	// DEBUG 
		float					m_vMFAngle;		// DEBUG

		Array<int>				m_stack;		// to avoid repeated mallocs
		Array<ReconSample>		m_reconSamples;	// to avoid repeated mallocs
	};

public:
	struct Sample
	{
		inline Vec4f		getTangentPlane(float time) const	{ return Vec4f(sec_normal, -dot(sec_normal,getHitPoint(time))); }	// @ secondary hit point
		inline Vec3f		getHitPoint(float time) const		{ return (s_motionEnabled) ? sec_hitpoint + (time-t)*sec_mv : sec_hitpoint; }

		Vec2f	xy;
		float	t;
		Vec3f	color;
//		Vec3f	pri_mv;			// Dof-motion test is retrofitted to the framework, and uses sec_mv instead.
		Vec3f	pri_normal;
		Vec3f	pri_albedo;
		Vec3f	sec_origin;
		Vec3f	sec_hitpoint;
		Vec3f	sec_mv;
		Vec3f	sec_normal;
		Vec3f   sec_albedo;
		Vec3f   sec_direct;

		Vec3i	origIndex;		// for experimentation, indexes input samplebuffer

		float	radius;
		static bool s_motionEnabled;
	};

	// for piping reconstruction rays from PBRT
	#pragma pack(push, 1)
	struct PBRTReconstructionRay
	{
		UnalignedVec2f	xy;
		Vec3f			o;
		Vec3f			d;
		Vec3f			weight;
	};
	#pragma pack(pop)

private:

	Array64<PBRTReconstructionRay>	m_PBRTReconstructionRays;
	Array<int>						m_PBRTReconstructionRaysScanlineStart;

	struct ReconSample
	{
		bool	backface;
		float	zdist;			// from x-plane
		Vec3f	color;			// here in order to ease certain debug visualizations
		float	weight;			// for filtering
		int		index;			// in m_samples
	};

	void fetchSample(Sample& s, const Vec3i& index) const
	{
		const int x=index[0], y=index[1], i=index[2];
		s.xy			= m_sbuf->getSampleXY	(x,y,i);
		s.t				= m_sbuf->getSampleT	(x,y,i);
		s.color			= m_sbuf->getSampleColor(x,y,i).getXYZ();
//		s.pri_mv		= m_sbuf->getSampleMV	(x,y,i);
		s.pri_normal	= m_sbuf->getSampleExtra<Vec3f>(CID_PRI_NORMAL  ,x,y,i);
		s.pri_albedo	= m_sbuf->getSampleExtra<Vec3f>(CID_ALBEDO      ,x,y,i);
		s.sec_origin	= m_sbuf->getSampleExtra<Vec3f>(CID_SEC_ORIGIN  ,x,y,i);
		s.sec_hitpoint	= m_sbuf->getSampleExtra<Vec3f>(CID_SEC_HITPOINT,x,y,i);
		s.sec_mv		= m_sbuf->getSampleExtra<Vec3f>(CID_SEC_MV      ,x,y,i);
		s.sec_normal	= m_sbuf->getSampleExtra<Vec3f>(CID_SEC_NORMAL  ,x,y,i);
		s.sec_albedo    = m_sbuf->getSampleExtra<Vec3f>(CID_SEC_ALBEDO  ,x,y,i);
		s.sec_direct    = m_sbuf->getSampleExtra<Vec3f>(CID_SEC_DIRECT  ,x,y,i);
		s.origIndex		= index;
	}

	Array<Sample>	m_samples;
	Array<Node>		m_hierarchy;		// root @ index 0

	int m_totalNumSamples;
	int	m_totalNumLeafNodes;
	int m_totalNumMixedLeaf;

	const UVTSampleBuffer* m_sbuf;
	int		m_numReconstructionRays;
	float	m_aoLength;					// >0 enables AO
	String	m_rayDumpFileName;
	bool	m_selectNearestSample;
	bool	m_useBandwidthInformation;
	bool	m_useDofMotionReconstruction;
	Vec4i	m_scissor;
};

}; // FW
