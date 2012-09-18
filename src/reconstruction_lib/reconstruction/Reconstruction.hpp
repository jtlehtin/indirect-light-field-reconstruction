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
#include "../common/SampleBuffer.hpp"
#include "../common/Util.hpp"
#include "base/MulticoreLauncher.hpp"
#include "base/Sort.hpp"
#include <cfloat>
#include <cstdio>


namespace FW
{

class Reconstruction
{
public:

	// Lehtinen et al. Siggraph 2012
	void	reconstructIndirect			(const UVTSampleBuffer& sbuf, int numReconstructionRays, Image& image, Image* debugImage=NULL, Vec4i scissor=Vec4i(0));	// scissor x0,y0,x1,y1; 0=inc, 1=exc
	void	reconstructIndirectCuda		(const UVTSampleBuffer& sbuf, int numReconstructionRays, Image& image);

	void	reconstructAO				(const UVTSampleBuffer& sbuf, int numReconstructionRays, float aoLength, Image& image, Image* debugImage=NULL, Vec4i scissor=Vec4i(0));
	void	reconstructAOCuda			(const UVTSampleBuffer& sbuf, int numReconstructionRays, float aoLength, Image& image);

	void	reconstructGlossy			(const UVTSampleBuffer& sbuf, String rayDumpFileName, Image& image, Image* debugImage=NULL, Vec4i scissor=Vec4i(0));
	void	reconstructGlossyCuda		(const UVTSampleBuffer& sbuf, String rayDumpFileName, Image& image);

	void	reconstructDofMotion		(const UVTSampleBuffer& sbuf, int numReconstructionRays, Image& image, Image* debugImage=NULL, Vec4i scissor=Vec4i(0));

	// RPF
	void	reconstructRPF				(const UVTSampleBuffer& sbuf, Image& image, Image* debugImage=NULL, float aoLength=0.f);

	// A-trous
	void	reconstructATrous			(const UVTSampleBuffer& sbuf, Image& image, Image* debugImage=NULL, float aoLength=0.f);
};



} //
