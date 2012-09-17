#pragma once
#include "base/Math.hpp"
#include "base/Random.hpp"

namespace FW
{
typedef Vector<F32,2> UnalignedVec2f;
typedef Vector<F32,3> UnalignedVec3f;
typedef Vector<F32,4> UnalignedVec4f;

Vec2f squareToDisk				(const Vec2f& v, bool Shirley=true);// [0,1] -> [-1,1]
Vec2f diskToSquare				(const Vec2f& v);					// [-1,1] -> [0,1]	(Uses Shirley)
Vec3f squareToUniformHemisphere	(const Vec2f& v);
Vec3f diskToUniformHemisphere	(const Vec2f& v);
Vec3f squareToCosineHemisphere	(const Vec2f& v, bool Shirley=true);
Vec3f diskToCosineHemisphere	(const Vec2f& v);

inline void CranleyPatterson	(float& val, float offset)				{ val += offset; if(val>=1) val-=1; }
inline void CranleyPatterson	(Vec2f& square, const Vec2f& offset)	{ square += offset; if(square[0]>=1) square[0]-=1; if(square[1]>=1) square[1]-=1; }

Vec2f ToUnitDisk				(const Vec2f& onSquare);			// [0,1] -> [-1,1]	Uses Shirley's low-distortion mapping
Vec2f FromUnitDisk				(const Vec2f& onDisk);				// [-1,1] -> [0,1]	Uses Shirley's low-distortion mapping	[Legacy]

float hammersley				(int i, int num);		// 0 <= i < n
float halton					(int base, int i);
float sobol						(int dim, const int i);	// 0 <= dim <=4

Vec2f sobol2D					(int i);
float larcherPillichshammer		(int i,U32 randomScramble=0);

U64   morton					(U32 x,U32 y);
U64   morton					(U32 x,U32 y,U32 z);

// given UNIT vector v, fills m with matrix whose 3rd column is v, and columns 0,1 are orthogonal to v and each other
// (i.e., returns matrix that maps from the local coordinates oriented with v into the ambient coordinates in which v is defined.
//  to go the other direction, i.e., ambient coordinates to the v-oriented local coordinate system, invert the resulting matrix).
Mat3f orthogonalBasis			(const Vec3f& v );
// intersects line o+t*d with plane, returns point. line is considered infinite.
Vec3f intersectRayPlane			(float& t, const Vec3f& o, const Vec3f& d, const Vec4f& plane );
inline Vec3f intersectRayPlane	(const Vec3f& o, const Vec3f& d, const Vec4f& plane ) { float t; return intersectRayPlane(t, o,d,plane); }

inline Vec3f reflect			(const Vec3f& I,const Vec3f& N)		{ return I - 2*dot(N,I)*N; }
inline Vec3f getXYW				(const Vec4f v)						{ return Vec3f(v.x,v.y,v.w); }

template <class T>
void permute(Random& random, Array<T>& data, int lo, int hi)
{
	const int N = hi-lo;
	for(int j=N;j>=2;j--)					// Permutation algo from wikipedia
	{
		int a = random.getS32(j);
		int b = j-1;
		swap(data[lo+a],data[lo+b]);
	}
}

inline bool isPowerOfTwo( U32 v )	{ return (v&(v-1)) == 0; }
inline U32  roundUpToNearestPowerOfTwo( U32 v )
{
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      return( v + 1 );
}
inline U32 log2(U32 v)
{
	const U32 b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
	const U32 S[] = {1, 2, 4, 8, 16};

	U32 r = 0;					// result of log2(v) will go here
	for(int i=4;i>=0;i--)
	{
		if (v & b[i])
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

class InsideConvexHull
{
public:
	InsideConvexHull() : wbox(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX) {}

	void add(const Vec2f& p, float s)
	{
		// +
		add(p+Vec2f(-s, 0));
		add(p+Vec2f( s, 0));
		add(p+Vec2f( 0,-s));
		add(p+Vec2f( 0, s));

		// x
		s /= sqrt(2.f);
		add(p+Vec2f(-s, s));
		add(p+Vec2f( s, s));
		add(p+Vec2f(-s,-s));
		add(p+Vec2f( s,-s));
	}

	void add(const Vec2f& p)
	{
		float k = p.x/p.y;
		if (p.y >= 0.f)	wbox[0] = min(wbox[0], k),	wbox[1] = max(wbox[1], k);
		else			wbox[2] = min(wbox[2], k),	wbox[3] = max(wbox[3], k);
	}

	bool originInside() const
	{
		return (wbox[1] > wbox[2] && wbox[0] < wbox[3]);
	}

private:
	Vec4f	wbox;
};



} //