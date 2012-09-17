#pragma once
#include "Util.hpp"
#include <stdio.h>

namespace FW
{

//------------------------------------------------------------------------

// Low-distortion map between square and circle (Shirley et al. JGT97)

Vec2f ToUnitDisk( const Vec2f& onSquare ) 
{
	const float PI = 3.1415926535897932384626433832795f;
	float phi, r, u, v;
	float a = 2*onSquare.x-1;
	float b = 2*onSquare.y-1;

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
	u = r * (float)cos( phi );
	v = r * (float)sin( phi );
	return Vec2f( u,v );
}

Vec2f FromUnitDisk(const Vec2f& onDisk)
{
	const float PI = 3.1415926535897932384626433832795f;
	float r   = sqrtf(onDisk.x * onDisk.x + onDisk.y * onDisk.y);
	float phi = atan2(onDisk.y, onDisk.x);

	if (phi < -PI/4)
		phi += 2*PI;

	float a, b, x, y;
	if (phi < PI/4)
	{
		a = r;
		b = phi * a / (PI/4);
	}
	else if (phi < 3*PI/4)
	{
		b = r;
		a = -(phi - PI/2) * b / (PI/4);
	}
	else if (phi < 5*PI/4)
	{
		a = -r;
		b = (phi - PI) * a / (PI/4);
	}
	else
	{
		b = -r;
		a = -(phi - 3*PI/2) * b / (PI/4);
	}

	x = (a + 1) / 2;
	y = (b + 1) / 2;
	return Vec2f(x, y);
}

Vec2f squareToDisk				(const Vec2f& square, bool Shirley)
{
	if(Shirley)
	{
		return ToUnitDisk(square);
	}
	else
	{
		const float u1 = square[0];
		const float u2 = square[1];
		const float r = sqrt( u1 );
		const float t = 2*FW_PI*u2;
		return Vec2f(r*cos(t),r*sin(t));
	}
}

Vec2f diskToSquare				(const Vec2f& v)
{
	return FromUnitDisk(v);
}

Vec3f squareToUniformHemisphere	(const Vec2f& square)
{
	const float u1 = square[0];
	const float u2 = square[1];
	const float r = sqrt(1-u1*u1);
	const float t = 2*FW_PI*u2;
	const float x = r*cos(t);
	const float y = r*sin(t);
	const float z = u1;
	return Vec3f(x,y,z);							// normalized direction vector in unit coordinate system
}

Vec3f diskToUniformHemisphere	(const Vec2f& disk)
{
	return squareToCosineHemisphere( diskToSquare(disk),true );	// TODO: a more direct method?
}

Vec3f squareToCosineHemisphere	(const Vec2f& square, bool Shirley)
{
	return diskToCosineHemisphere( squareToDisk(square,Shirley) );
}

Vec3f diskToCosineHemisphere	(const Vec2f& disk)
{
	const float z = sqrt(1-dot(disk,disk));			// lift to hemisphere -> distributed according to cosine-weighting
	return Vec3f(disk,z);							// normalized direction vector in unit coordinate system
}

//------------------------------------------------------------------------

float hammersley(int i, int num)
{
	FW_ASSERT(i>=0 && i<num);
	return (i+0.5f) / num;
}

float halton(int base, int i)
{
	FW_ASSERT(i>=0);

	float h = 0.f;
	float half = rcp(float(base));

	while (i)
	{
		int digit = i % base;
		h = h + digit * half;
		i = (i-digit) / base;
		half = half / base;
	}

	return h;
}

unsigned int SobolGeneratingMatrices[] = {
#include "Sobol5.inl"	// From Leo/MI
};

float sobol(int dim, int i)
{
	FW_ASSERT(i >= 0 && dim >= 0 && dim < 5);
	const unsigned int* const matrix = SobolGeneratingMatrices + dim * 32;
	unsigned int result = 0;
	for (unsigned int c = 0; i; i >>= 1, ++c)
		if (i & 1)
			result ^= matrix[c];
	return result * (1.f / (1ULL << 32));
}

Vec2f sobol2D(int i)
{
	FW_ASSERT(i>=0);
	Vec2f result;
	// remaining components by matrix multiplication 
	unsigned int r1 = 0, r2 = 0; 
	for (unsigned int v1 = 1U << 31, v2 = 3U << 30; i; i >>= 1)
	{
		if (i & 1)
		{
			// vector addition of matrix column by XOR
			r1 ^= v1; 
			r2 ^= v2 << 1;
		}
		// update matrix columns 
		v1 |= v1 >> 1; 
		v2 ^= v2 >> 1;
	}
	// map to unit cube [0,1)^3
	result[0] = r1 * (1.f / (1ULL << 32));
	result[1] = r2 * (1.f / (1ULL << 32));
	return result;
}

float larcherPillichshammer(int i,U32 r)
{
	FW_ASSERT(i>=0);
	for(U32 v=1U<<31; i; i>>=1, v|=v>>1)
		if(i&1)
			r^=v;

	return float( (double)r / (double)0x100000000LL);
}

// --------------------------------------------------------------------------

Mat3f orthogonalBasis( const Vec3f& v )
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

// --------------------------------------------------------------------------

Vec3f intersectRayPlane(float& t, const Vec3f& o, const Vec3f& d, const Vec4f& plane)
{
	// (o+t*d,1) DOT plane = 0
	// t*d DOT plane.xyz = -(o,1) DOT plane
	// t = -(o,1) DOT plane / (d DOT plane.xyz)

	float ddp = dot( d, plane.getXYZ() );
	if (ddp==0)							// ray parallel to the plane
	{
		if(dot(plane,Vec4f(o,1))==0)	// ray on the plane
		{
			t = 0;
			return o;
		}
		else
		{
			t = FW_F32_MAX;
			return FW_F32_MAX;
		}
	}
	else
	{
		t = (-(dot(o,plane.getXYZ()) + plane.w) / ddp);
		return o + t*d;
	}
}

// --------------------------------------------------------------------------

U64 morton(U32 x,U32 y)
{
	U64 xx = x;
	xx = (xx | xx << 16) & 0x0000FFFF0000FFFFull;
	xx = (xx | xx << 8)  & 0x00FF00FF00FF00FFull;
	xx = (xx | xx << 4)  & 0x0F0F0F0F0F0F0F0Full;
	xx = (xx | xx << 2)  & 0x3333333333333333ull;
	xx = (xx | xx << 1)  & 0x5555555555555555ull;
	U64 yy = y;
	yy = (yy | yy << 16) & 0x0000FFFF0000FFFFull;
	yy = (yy | yy << 8)  & 0x00FF00FF00FF00FFull;
	yy = (yy | yy << 4)  & 0x0F0F0F0F0F0F0F0Full;
	yy = (yy | yy << 2)  & 0x3333333333333333ull;
	yy = (yy | yy << 1)  & 0x5555555555555555ull;

	return xx | (yy << 1);
}

U64 morton(U32 x,U32 y,U32 z)
{
	U64 xx = x;
	xx = (xx | xx << 32) & 0xFFFF00000000FFFFull;
	xx = (xx | xx << 16) & 0x00FF0000FF0000FFull;
	xx = (xx | xx << 8)  & 0xF00F00F00F00F00Full;
	xx = (xx | xx << 4)  & 0x30C30C30C30C30C3ull;
	xx = (xx | xx << 2)  & 0x9249249249249249ull;
	U64 yy = y;
	yy = (yy | yy << 32) & 0xFFFF00000000FFFFull;
	yy = (yy | yy << 16) & 0x00FF0000FF0000FFull;
	yy = (yy | yy << 8)  & 0xF00F00F00F00F00Full;
	yy = (yy | yy << 4)  & 0x30C30C30C30C30C3ull;
	yy = (yy | yy << 2)  & 0x9249249249249249ull;
	U64 zz = z;
	zz = (zz | zz << 32) & 0xFFFF00000000FFFFull;
	zz = (zz | zz << 16) & 0x00FF0000FF0000FFull;
	zz = (zz | zz << 8)  & 0xF00F00F00F00F00Full;
	zz = (zz | zz << 4)  & 0x30C30C30C30C30C3ull;
	zz = (zz | zz << 2)  & 0x9249249249249249ull;
	return (xx) | (yy<<1) | (zz<<2);
}

} //