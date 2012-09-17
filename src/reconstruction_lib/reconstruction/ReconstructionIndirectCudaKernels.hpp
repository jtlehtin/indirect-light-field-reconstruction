//-------------------------------------------------------------------------------------------------
// CUDA indirect reconstruction kernel headers
//-------------------------------------------------------------------------------------------------

#pragma once
#include "base/DLLImports.hpp"
#include "base/Math.hpp"

namespace FW
{

struct CudaReceiverInd
{
	int		pixel;
	Vec3f	pos;
	Vec3f	normal;
	Vec3f	albedo;
};

struct CudaSampleInd
{
	Vec3f	pos;
	Vec3f	normal;
	Vec3f	color;
	Vec3f   orig;
	float   size;		// negative size means this is the last sample of node
	float   plen;
	float   bw;
};

struct CudaTSampleInd
{
	Vec4f   posSize;
	Vec4f   normalPlen;
};

struct CudaNodeInd
{
	int		idx0, idx1;
	Vec3f	bbmin[2];
	Vec3f	bbmax[2];
};

struct CudaTNodeInd
{
	Vec4f   hdr;
	Vec4f   n0xy;
	Vec4f   n1xy;
	Vec4f   nz;
};

struct CudaShrinkRayInd
{
	Vec3f   origin;
	Vec3f   endpoint;
};

struct CudaPBRTRay
{
	int     pixel;
	Vec3f	o;
	Vec3f	d;
	Vec3f	weight;
};

//-------------------------------------------------------------------------------------------------

struct IndirectKernelInput
{
	CUdeviceptr     recv;
	CUdeviceptr		nodes;
	CUdeviceptr		samples;
	CUdeviceptr		gnodes;	  // geom hierarchy for two-pass glossy
	CUdeviceptr     sobol;
	CUdeviceptr		resultImg;
	CUdeviceptr     pbrtRays;
	CUdeviceptr     rays;  // for shrink kernel
	CUdeviceptr     radii; // for shrink kernel
	int2			size;
	U32             nr;
	U32             pbrtRayCount;
	U32             outputSpp;
	U32				firstRay;
	U32				numRays;
};

#if FW_CUDA
extern "C"
{
	__constant__ IndirectKernelInput c_IndirectKernelInput;
	__device__ S32 g_overflowCount;
	__device__ S32 g_emptyCount;
	texture<float4, 1> t_nodes;
	texture<float4, 1> t_samples;
	texture<float4, 1> t_gnodes;
}
#endif

//-------------------------------------------------------------------------------------------------
}
