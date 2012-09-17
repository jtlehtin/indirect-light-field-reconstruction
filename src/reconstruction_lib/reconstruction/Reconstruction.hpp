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
