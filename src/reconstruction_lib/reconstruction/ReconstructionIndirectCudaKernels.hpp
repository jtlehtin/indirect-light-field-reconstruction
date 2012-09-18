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
