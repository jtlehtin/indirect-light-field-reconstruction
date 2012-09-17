//-------------------------------------------------------------------------------------------------
// CUDA reconstruction kernels
//-------------------------------------------------------------------------------------------------

#include "ReconstructionIndirectCudaKernels.hpp"
#include <stdio.h>

enum
{
	STACK_SIZE          = 64,
	MAX_SAMPLES         = 512,
	SMALL_SURFACE_LIMIT = 4,
	SENTINEL            = 0x76543210,
};

#define HULL_TYPE Vec4f
#define HULL_EMPTY (Vec4f(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX))
#define HULL_ADD(abox,v) do {\
	float dx = (v).x;\
	float dy = (v).y;\
	float k = dx/dy;\
	if (dy >= 0.f)	abox.x = fmin(abox.x, k), abox.y = fmax(abox.y, k);\
	else			abox.z = fmin(abox.z, k), abox.w = fmax(abox.w, k);\
	} while (0)
#define HULL_INSIDE(abox) (abox.y > abox.z && abox.x < abox.w)

using namespace FW;

//------------------------------------------------------------------------

__device__ __inline__ U32   getLo                   (U64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ S32   getLo                   (S64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ U32   getHi                   (U64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ S32   getHi                   (S64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ U64   combineLoHi             (U32 lo, U32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ S64   combineLoHi             (S32 lo, S32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ U32   getLaneMaskLt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskLe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_le;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_gt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_ge;" : "=r"(r)); return r; }
__device__ __inline__ int   findLeadingOne          (U32 v)                 { U32 r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
__device__ __inline__ int   findLastOne             (U32 v)                 { return 31-findLeadingOne(__brev(v)); } // 0..31, 32 for not found
__device__ __inline__ bool  singleLane              (void)                  { return ((__ballot(true) & getLaneMaskLt()) == 0); }
__device__ __inline__ U32   rol						(U32 x, U32 s)			{ return (x<<s)|(x>>(32-s)); }
__device__ __inline__ Vec2f U64toVec2f				(U64 xy)				{ return Vec2f(__int_as_float(getLo(xy)), __int_as_float(getHi(xy))); }
__device__ __inline__ U64   Vec2ftoU64				(const Vec2f& v)		{ return combineLoHi(__float_as_int(v.x), __float_as_int(v.y)); }
__device__ __inline__ int   imin					(S32 a, S32 b)			{ S32 v; asm("min.s32 %0, %1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ int   imax					(S32 a, S32 b)			{ S32 v; asm("max.s32 %0, %1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }

//------------------------------------------------------------------------

#define FW_HASH_MAGIC   (0x9e3779b9u)
#define FW_JENKINS_MIX(a, b, c)   \
    a -= b; a -= c; a ^= (c>>13); \
    b -= c; b -= a; b ^= (a<<8);  \
    c -= a; c -= b; c ^= (b>>13); \
    a -= b; a -= c; a ^= (c>>12); \
    b -= c; b -= a; b ^= (a<<16); \
    c -= a; c -= b; c ^= (b>>5);  \
    a -= b; a -= c; a ^= (c>>3);  \
    b -= c; b -= a; b ^= (a<<10); \
    c -= a; c -= b; c ^= (b>>15);

__device__ __inline__ U32  hashBits(U32 a, U32 b = FW_HASH_MAGIC, U32 c = 0)
{
	c += FW_HASH_MAGIC;
	FW_JENKINS_MIX(a, b, c);
	return c;
}

//------------------------------------------------------------------------

__device__ Mat3f orthogonalBasis(const Vec3f& v)
{
	Mat3f m;
	Vec3f mx = v;
	if( fabs( mx.x ) > fabs( mx.y ) && fabs( mx.x ) > fabs( mx.z ) )
	{
		FW::swap( mx.x, mx.y );
		mx.x = -mx.x;
	}
	else if( fabs( mx.y ) > fabs( mx.x ) && fabs( mx.y ) > fabs( mx.z ) )
	{
		FW::swap( mx.y, mx.z );
		mx.y = -mx.y;
	}
	else 
	{
		FW::swap( mx.z, mx.x );
		mx.z = -mx.z;
	}
	m.setCol( 1, cross(v, mx).normalized() );
	m.setCol( 0, cross(m.getCol(1), v).normalized() );
	m.setCol( 2, v );
	return m;
}

__device__ Vec2f toUnitDisk(const Vec2f& onSquare) 
{
	const float PI = 3.1415926535897932384626433832795f;
	float phi, r, u, v;
	float a = 2.f * onSquare.x - 1.f;
	float b = 2.f * onSquare.y - 1.f;

	if (a > -b)
	{
		if (a > b)
		{
			r=a;
			phi = (PI/4 ) * (b/a);
		}
		else
		{
			r = b;
			phi = (PI/4) * (2 - (a/b));
		}
	}
	else
	{
		if (a < b)
		{
			r = -a;
			phi = (PI/4) * (4 + (b/a));
		}
		else
		{
			r = -b;
			if (b != 0)	phi = (PI/4) * (6 - (a/b));
			else		phi = 0;
		}
	}
	u = r * (float)cosf(phi);
	v = r * (float)sinf(phi);
	return Vec2f(u, v);
}

__device__ Vec3f diskToCosineHemisphere(const Vec2f& disk)
{
	return Vec3f(disk.x, disk.y, sqrtf(fabsf(1.f - dot(disk, disk))));
}

__device__ Vec3f squareToCosineHemisphere(const Vec2f& square)
{
	return diskToCosineHemisphere(toUnitDisk(square));
}

__device__ float fminf(float a, float b, float c)				{ return fminf(fminf(a, b), c); }
__device__ float fminf(float a, float b, float c, float d)		{ return fminf(fminf(fminf(a, b), c), d); }
__device__ float fmaxf(float a, float b, float c)				{ return fmaxf(fmaxf(a, b), c); }
__device__ float fmaxf(float a, float b, float c, float d)		{ return fmaxf(fmaxf(fmaxf(a, b), c), d); }

__device__ 	bool insideConvexHull(U64* samples, int lo, int hi, Vec3f& origin, Mat3f& basis)
{
	if (hi - lo > 3)
		return true;

	float Rscale;
	switch(hi - lo)
	{
		case 1:	Rscale = 0.5f; break;
		case 2:	Rscale = 0.6f; break;
		case 3:	Rscale = 0.7f; break;
	}

	HULL_TYPE hull = HULL_EMPTY;
	for (int i=lo; i < hi; i++)	
	{
		U64 st = samples[MAX_SAMPLES - i - 1];

#if 1
		int sidx = getLo(st) * 2;
		float4 td0 = tex1Dfetch(t_samples, sidx + 0);
		float4 td1 = tex1Dfetch(t_samples, sidx + 1);
		Vec3f p = Vec3f(td0.x, td0.y, td0.z) - origin;
		Vec3f n = Vec3f(td1.x, td1.y, td1.z);
		float R = fabsf(td0.w);
#else
		CudaSampleInd& s = ((CudaSampleInd*)in.samples)[getLo(st)];
		Vec3f p = s.pos - origin;
		Vec3f n = s.normal;
		float R = fabsf(s.size);
#endif

		Vec3f P = basis * p; // splat center in camera space
		Vec3f N = basis * n; // splat normal in camera space
		float invw = 1.f / P.z;
		R *= invw;

		const float cosAngle = fabsf(N.z);
		Vec2f xy  = P.getXY() * invw;
		Vec2f N2d = N.getXY().normalized();
		if((__float_as_int(N2d.x) | __float_as_int(N2d.y)) == 0)
			N2d.x = 1.f;

		const float minorScale = Rscale * R * cosAngle;
		const float majorScale = Rscale * R;
		const Vec2f minorAxis  = minorScale * Vec2f(N2d.x,  N2d.y);
		const Vec2f majorAxis  = majorScale * Vec2f(N2d.y, -N2d.x);

		HULL_ADD(hull, xy + minorAxis);
		HULL_ADD(hull, xy - minorAxis);
		HULL_ADD(hull, xy + majorAxis);
		HULL_ADD(hull, xy - majorAxis);
		
		const float diagScale = .5f * Rscale * R * sqrtf(1 + cosAngle*cosAngle);
		const Vec2f diag1Axis = diagScale * Vec2f(N2d.x + N2d.y, N2d.y - N2d.x);
		const Vec2f diag2Axis = diagScale * Vec2f(N2d.x - N2d.y, N2d.y + N2d.x);

		HULL_ADD(hull, xy + diag1Axis);
		HULL_ADD(hull, xy - diag1Axis);
		HULL_ADD(hull, xy + diag2Axis);
		HULL_ADD(hull, xy - diag2Axis);
		
		if (HULL_INSIDE(hull))
			return true;
	}

	return false;
}

//------------------------------------------------------------------------

#define storeResult(idx_,color_) do {\
	float* cptr = (float*)(in.resultImg + ((idx_) << 4)); \
	atomicAdd(&cptr[0], (color_).x); \
	atomicAdd(&cptr[1], (color_).y); \
	atomicAdd(&cptr[2], (color_).z); \
	atomicAdd(&cptr[3], (color_).w); \
} while (0)

__device__ void accumulateColor(Vec4f& color, const Vec4f& c)
{
#ifdef SELECT_NEAREST_SAMPLE
	if (c.w > color.w)
		color = c;
#else
	color += c;
#endif
}

__device__ float vMFfromBandwidth(float bw)
{
	return 4.0f * sqrtf(bw);
}

//------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(128,6) filterKernel(void)
{
    const IndirectKernelInput& in = *(const IndirectKernelInput*)&c_IndirectKernelInput;

    int tidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if (tidx >= in.numRays)
		return;

	const float t_eps   = 1e-3f;
	const float ray_eps = 1e-20f;

	int reconRayIdx = tidx + in.firstRay;
	int receiverIdx = reconRayIdx / in.nr;
	int rayPixelIdx = reconRayIdx % in.outputSpp;

	int   pixel;
	Vec3f origin;
	Vec3f direction;
	Vec3f weight;

	if (in.pbrtRayCount > 0)
	{
		CudaPBRTRay* ray = &((CudaPBRTRay*)in.pbrtRays)[tidx];

		pixel     = ray->pixel;
		origin    = ray->o;
		direction = ray->d;
		weight    = ray->weight;

		if (direction.isZero() || weight.isZero())
		{
			Vec4f black(0,0,0,1);
			storeResult(pixel, black);
			return;
		}
	} else
	{
		// construct reconstruction ray

		CudaReceiverInd* recv = &((CudaReceiverInd*)in.recv)[receiverIdx];

		pixel = recv->pixel;
		Vec2f dsqr = ((Vec2f*)in.sobol)[rayPixelIdx];
		dsqr.x += hashBits(hashBits(pixel, FW_HASH_MAGIC, 0)) * (1.f / FW_U32_MAX);
		dsqr.y += hashBits(hashBits(pixel, FW_HASH_MAGIC, 1)) * (1.f / FW_U32_MAX);
		if (dsqr.x >= 1.f) dsqr.x -= 1.f;
		if (dsqr.y >= 1.f) dsqr.y -= 1.f;
		Vec3f dunit     = squareToCosineHemisphere(dsqr);

		origin    = recv->pos + recv->normal * t_eps;
		direction = (orthogonalBasis(recv->normal) * dunit).normalized();
		weight    = recv->albedo;
	}

	Mat3f basis = orthogonalBasis(direction).transposed();

	// traversal stack
	int stack[STACK_SIZE];
	stack[0] = SENTINEL; // Bottom-most entry.
	char* stackPtr = (char*)&stack[0];

	// sample array
	U64 samples[MAX_SAMPLES];	// lo = idx, hi = key
	int numSamples = 0;

	// ray parameters
	Vec3f idir;
	idir.x = 1.0f / (fabs(direction.x) > ray_eps ? direction.x : (direction.x < 0 ? -ray_eps : ray_eps));
	idir.y = 1.0f / (fabs(direction.y) > ray_eps ? direction.y : (direction.y < 0 ? -ray_eps : ray_eps));
	idir.z = 1.0f / (fabs(direction.z) > ray_eps ? direction.z : (direction.z < 0 ? -ray_eps : ray_eps));
	Vec3f ood = origin * idir;

	// todo: heapify?

	int nodeAddr = 0;
	int leafAddr = 0;
	while(nodeAddr != SENTINEL)
	{
		bool searchingLeaf = true;
		while (nodeAddr >= 0 && nodeAddr != SENTINEL)
		{
			// internal node, intersect against child nodes
			int idx = nodeAddr;
#if 1
			float4 hdr  = tex1Dfetch(t_nodes, idx*4);
			float4 n0xy = tex1Dfetch(t_nodes, idx*4+1);
			float4 n1xy = tex1Dfetch(t_nodes, idx*4+2);
			float4 nz   = tex1Dfetch(t_nodes, idx*4+3);
			int idx0 = __float_as_int(hdr.x);
			int idx1 = __float_as_int(hdr.y);
#else
			CudaNodeInd& node = ((CudaNodeInd*)in.nodes)[idx];
			int idx0 = node.idx0;
			int idx1 = node.idx1;
			float4 n0xy = make_float4(node.bbmin[0].x, node.bbmax[0].x, node.bbmin[0].y, node.bbmax[0].y);
			float4 n1xy = make_float4(node.bbmin[1].x, node.bbmax[1].x, node.bbmin[1].y, node.bbmax[1].y);
			float4 nz   = make_float4(node.bbmin[0].z, node.bbmax[0].z, node.bbmin[1].z, node.bbmax[1].z);
#endif
			float c0lox = n0xy.x * idir.x - ood.x;
			float c0hix = n0xy.y * idir.x - ood.x;
			float c0loy = n0xy.z * idir.y - ood.y;
			float c0hiy = n0xy.w * idir.y - ood.y;
			float c0loz = nz.x   * idir.z - ood.z;
			float c0hiz = nz.y   * idir.z - ood.z;
			float c1loz = nz.z   * idir.z - ood.z;
			float c1hiz = nz.w   * idir.z - ood.z;
			float c0min = fmaxf(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), 0.f);
			float c0max = fminf(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz));
			float c1lox = n1xy.x * idir.x - ood.x;
			float c1hix = n1xy.y * idir.x - ood.x;
			float c1loy = n1xy.z * idir.y - ood.y;
			float c1hiy = n1xy.w * idir.y - ood.y;
			float c1min = fmaxf(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), 0.f);
			float c1max = fminf(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz));

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			if (!traverseChild0 && !traverseChild1)
			{
				// Neither child was intersected => pop stack.
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
			else
			{
				// Otherwise fetch child pointers.
				// todo postpone fetch until here
				nodeAddr = (traverseChild0) ? idx0 : idx1;

				// Both children were intersected => push one.
				if (traverseChild0 && traverseChild1)
				{
					if (c1min < c0min)
						swap(nodeAddr, idx1);
					stackPtr += 4;
					*(int*)stackPtr = idx1;
				}
			}

			// First leaf => postpone and continue traversal.
			if (nodeAddr < 0 && leafAddr >= 0)
			{
				searchingLeaf = false;
				leafAddr = nodeAddr;
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}

			// All SIMD lanes have found a leaf => process them.
			if (!__any(searchingLeaf))
				break;
		}

		// Process postponed leaf nodes.

		while (leafAddr < 0)
		{
			// leaf node, test against samples here
			float ssize = 0.f;
			for (int i = ~leafAddr; ssize >= 0.f; i++)
			{
#if 1
				int sidx = i * 2;
				float4 td0 = tex1Dfetch(t_samples, sidx+0);
				float4 td1 = tex1Dfetch(t_samples, sidx+1);
				Vec3f spos(td0.x, td0.y, td0.z);
				Vec3f snormal(td1.x, td1.y, td1.z);
				float splen = td1.w;
				ssize = td0.w;
#else
				CudaSampleInd& s = ((CudaSampleInd*)in.samples)[i];
				Vec3f spos = s.pos;
				Vec3f snormal = s.normal;
				float splen = s.plen;
				ssize = s.size;
#endif
				// distance to splat plane along ray direction
				Vec3f y  = spos - origin;
				float yn = dot(y, snormal);			// if positive, splat is back-facing
				float dn = dot(direction, snormal);	// if positive, splat is back-facing
				float t = yn * (1.f / dn);
				if (t < t_eps * splen) // epsilon based on ray length
					continue; // avoid hitting the originating surface

				// ray intersection on splat plane
				Vec3f p = origin + t * direction;
				if ((p - spos).lenSqr() > ssize * ssize)
					continue; // does not hit the splat

				// distance to splat center from ray normal plane
				float dist_st = dot(y, direction);

				// back-face in the nearfield -> cull (basically because small concavities suffer from undersampling)
				if (yn >= 0.f && dist_st < fabsf(ssize))
					continue;

				// rare but possible
				if (numSamples == MAX_SAMPLES)
				{
					atomicAdd(&g_overflowCount, 1);
					return;
				}

				// add sample into a binary heap
				int x = numSamples;
				while (x > 0)
				{
					int   p = ((x+1) >> 1) - 1;
					U64   sp = samples[p];
					float tp = __int_as_float(getHi(sp));
					if (tp > t)
						samples[x] = sp;
					else
						break;
					x = p;
				}
				samples[x] = combineLoHi(i, __float_as_int(t));
				numSamples++;
			}

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
		} // leaf
	} // traversal

	// no samples? terminate
	if (!numSamples)
	{
		atomicAdd(&g_emptyCount, 1);
		return;
	}

	// process in near->far order
	Vec4f color = 0.f;
	int heapsize = numSamples;
	int first      = 0; // first sample of the current surface
	int firstaccum = 0; // first sample we have accumulated so far
	int idx        = 0;
	for (; idx < numSamples; idx++)
	{
		// extract top of heap
		U64 st = samples[0];

		// pop last and restore heap property
		heapsize--;
		if (heapsize)
		{
			U64   sb = samples[heapsize];
			float tb = __int_as_float(getHi(sb));
			int   x  = 0;
			int   c0 = ((x+1)*2)-1;
			int   c1 = c0 + 1;
			while (c0 < heapsize)
			{
				U64   sc0 = samples[c0];
				U64   sc1 = c1 < heapsize ? samples[c1] : combineLoHi(0, __float_as_int(FW_F32_MAX));
				float tc0 = __int_as_float(getHi(sc0));
				float tc1 = __int_as_float(getHi(sc1));
				if (fminf(tc0, tc1) < tb)
				{
					bool min0 = (tc0 < tc1);
					samples[x] = min0 ? sc0 : sc1;
					x = min0 ? c0 : c1;
					c0 = 2*x + 1;
					c1 = c0 + 1;
				} else
					break;
			}
			samples[x] = sb;
		}

		// store sample
		samples[MAX_SAMPLES - idx - 1] = st;

		// get sample data
#if 0
		// debug debug this is broken! no color
		int            sidx   = getLo(st) * 2;
		float          t      = __int_as_float(getHi(st));
		float4         td0    = tex1Dfetch(t_samples, sidx+0);
		float4         td1    = tex1Dfetch(t_samples, sidx+1);
		float4         td2    = tex1Dfetch(t_samples, sidx+2);
		Vec3f          n      (td1.x, td1.y, td1.z);
		Vec3f          spos   (td0.x, td0.y, td0.z);
		float          ssize  = fabsf(td0.w);
		Vec3f          scolor (td2.x, td2.y, td2.z);
#else
		CudaSampleInd& s      = ((CudaSampleInd*)in.samples)[getLo(st)];
		float          t      = __int_as_float(getHi(st));
		Vec3f          n      = s.normal;
		Vec3f          spos   = s.pos;
		float          ssize  = fabsf(s.size);
		Vec3f          scolor = s.color;
#endif

		Vec4f c(0.f); // color to be accumulated
		if (dot(n, direction) >= 0)
		{
			n = -n;
		} else
		{
			// calculate color to be accumulated
			#ifdef AMBIENT_OCCLUSION
			{
				c = (t < AMBIENT_OCCLUSION) ? 0.f : 1.f;
				c.w = 1.f;
			}
			#else
			{
				Vec3f p = origin + t * direction;

				#ifdef USE_BANDWIDTH_INFORMATION
				{
					float anglecos = dot((s.orig - spos).normalized(), -direction);
					float bw = vMFfromBandwidth(s.bw);
					float vMF = expf(bw * anglecos - bw); // vMF but normalized to [0,1]

					// How large is the splat compared to the length of the ray?
#if 0
					float distWeight = fmin(1.f, ssize / t);
					const float spatialWeight = fmax(0.0f, 1.0f - (p - spos).length() / ssize);
					float w = vMF * fmin(1.f, spatialWeight / distWeight);
#else
					// another way: distance on UV plane between rec.ray and ray to splat center (equally good)
					float d = 1.f / dot(direction, (spos - origin).normalized());
					d = .5f * sqrtf(fabsf(d*d - 1.f)); // times 0.5 for good measure
					float w = vMF * fmax(0.f, 1.f - d);
#endif
					c = Vec4f(w * scolor, w);
				}
				#else
				{
					float w = 1.f - (spos - p).length() / ssize;
					c = Vec4f(w * scolor, w);
				}
				#endif
			}
			#endif
		}

		// test against previous samples in surface
		bool conflict = false;
		for (int i=first; i < idx && !conflict; i++)
		{
			U64 st0 = samples[MAX_SAMPLES - i - 1];
			CudaSampleInd& s0 = ((CudaSampleInd*)in.samples)[getLo(st0)];
			float          t0 = __int_as_float(getHi(st0));
			Vec3f          n0 = s0.normal;
			if (dot(n0, direction) >= 0)
				n0 = -n0;

			// consistently facing (away from) each other?
			Vec3f diff = (spos - s0.pos);
			float invlen = rsqrtf(diff.lenSqr());
			float cosAngle1 =  dot(n0, diff);
			float cosAngle2 = -dot(n,  diff);
			const float eps = 0.035f;
			conflict = (fmaxf(fminf(cosAngle1, cosAngle2), -fmaxf(cosAngle1, cosAngle2)) * invlen < -eps);
		}

		// if there wasn't a conflict, accumulate and continue
		if (!conflict)
		{
			accumulateColor(color, c);
			continue;
		}

		// there was a conflict, always start a logically new surface
		first = idx;

		// if not enough samples, accumulate and continue
		if (idx - firstaccum < SMALL_SURFACE_LIMIT)
		{
			// but if the surface covers the sample in convex hull sense, use it!
			if (insideConvexHull(samples, firstaccum, idx, origin, basis))
				break;

			accumulateColor(color, c);
			continue;
		}

		// convex hull was okay, break and return result
		break;
	}

	// if nothing found, treat as black (e.g. only backfacing splats)
	if (color.w == 0.f)
		color.w = 1.f;

	// normalize color
	color *= 1.f/color.w;

#ifndef AMBIENT_OCCLUSION
	color.x *= weight.x;
	color.y *= weight.y;
	color.z *= weight.z;
#endif

	storeResult(pixel, color); 
}

//------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(128,6) shrinkKernel(void)
{
    const IndirectKernelInput& in = *(const IndirectKernelInput*)&c_IndirectKernelInput;

    int tidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if (tidx >= in.numRays)
		return;
	tidx += in.firstRay;

	const float ray_eps = 1e-20f;

	// fetch ray
	CudaShrinkRayInd& ray = ((CudaShrinkRayInd*)in.rays)[tidx];
	Vec3f origin    = ray.origin;
	Vec3f hitp      = ray.endpoint;
	Vec3f direction = hitp - origin;
	float rayLen = direction.length();
	direction *= 1.f / rayLen; // normalize

	// traversal stack
	int stack[STACK_SIZE];
	stack[0] = SENTINEL; // Bottom-most entry.
	char* stackPtr = (char*)&stack[0];

	// ray traversal parameters
	Vec3f idir;
	idir.x = 1.0f / (fabs(direction.x) > ray_eps ? direction.x : (direction.x < 0 ? -ray_eps : ray_eps));
	idir.y = 1.0f / (fabs(direction.y) > ray_eps ? direction.y : (direction.y < 0 ? -ray_eps : ray_eps));
	idir.z = 1.0f / (fabs(direction.z) > ray_eps ? direction.z : (direction.z < 0 ? -ray_eps : ray_eps));
	Vec3f ood = origin * idir;

	// adaptive epsilon similar to from PBRT (takes max because for defocus the rays start from the origin of camera space)
	float eps = 1e-3f * fmaxf(origin.length(), hitp.length());

	int nodeAddr = 0;
	int leafAddr = 0;
	while(nodeAddr != SENTINEL)
	{
		bool searchingLeaf = true;
		while (nodeAddr >= 0 && nodeAddr != SENTINEL)
		{
			// internal node, intersect against child nodes
			int idx = nodeAddr;
			float4 hdr  = tex1Dfetch(t_nodes, idx*4);
			float4 n0xy = tex1Dfetch(t_nodes, idx*4+1);
			float4 n1xy = tex1Dfetch(t_nodes, idx*4+2);
			float4 nz   = tex1Dfetch(t_nodes, idx*4+3);
			int idx0 = __float_as_int(hdr.x);
			int idx1 = __float_as_int(hdr.y);
			float c0lox = n0xy.x * idir.x - ood.x;
			float c0hix = n0xy.y * idir.x - ood.x;
			float c0loy = n0xy.z * idir.y - ood.y;
			float c0hiy = n0xy.w * idir.y - ood.y;
			float c0loz = nz.x   * idir.z - ood.z;
			float c0hiz = nz.y   * idir.z - ood.z;
			float c1loz = nz.z   * idir.z - ood.z;
			float c1hiz = nz.w   * idir.z - ood.z;
			float c0min = fmaxf(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), 0.f);
			float c0max = fminf(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz));
			float c1lox = n1xy.x * idir.x - ood.x;
			float c1hix = n1xy.y * idir.x - ood.x;
			float c1loy = n1xy.z * idir.y - ood.y;
			float c1hiy = n1xy.w * idir.y - ood.y;
			float c1min = fmaxf(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), 0.f);
			float c1max = fminf(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz));

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			if (!traverseChild0 && !traverseChild1)
			{
				// Neither child was intersected => pop stack.
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
			else
			{
				// Otherwise fetch child pointers.
				// todo postpone fetch until here
				nodeAddr = (traverseChild0) ? idx0 : idx1;

				// Both children were intersected => push one.
				if (traverseChild0 && traverseChild1)
				{
					if (c1min < c0min)
						swap(nodeAddr, idx1);
					stackPtr += 4;
					*(int*)stackPtr = idx1;
				}
			}

			// First leaf => postpone and continue traversal.
			if (nodeAddr < 0 && leafAddr >= 0)
			{
				searchingLeaf = false;
				leafAddr = nodeAddr;
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}

			// All SIMD lanes have found a leaf => process them.
			if (!__any(searchingLeaf))
				break;
		}

		// Process postponed leaf nodes.

		while (leafAddr < 0)
		{
			// leaf node, test against samples here
			float ssize = 0.f;
			for (int i = ~leafAddr; ssize >= 0.f; i++)
			{
				int sidx = i * 2;
				float4 td0 = tex1Dfetch(t_samples, sidx+0);
				float4 td1 = tex1Dfetch(t_samples, sidx+1);
				Vec3f spos(td0.x, td0.y, td0.z);
				Vec3f snormal(td1.x, td1.y, td1.z);
				ssize = td0.w;

				// distance to splat plane along ray direction
				Vec3f y  = spos - origin;
				float yn = dot(y, snormal);			// if positive, splat is back-facing
				float dn = dot(direction, snormal);	// if positive, splat is back-facing
				float t = yn * (1.f / dn);
				if (t <= eps || t >= rayLen - eps) // epsilon based on ray length
					continue; // avoid accidental hits very close to origin and hitpoint

				// ray intersection on splat plane
				Vec3f p = origin + t * direction;
				float tpdist2 = (p - spos).lenSqr();
				if (tpdist2 > ssize * ssize)
					continue; // does not hit the splat

				// shrink it
				atomicMin(&((int*)in.radii)[i], __float_as_int(sqrtf(tpdist2)));
			}

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
		} // leaf
	} // traversal
}

