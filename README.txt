***********************************************************************
*** An implementation of
***
*** Lehtinen, J., Aila, T., Laine, S., and Durand, F. 2012,
*** Reconstructing the Indirect Light Field for Global Illumination,
*** ACM Transactions on Graphics 31(4) (Proc. ACM SIGGRAPH 2012), article 51.
***
*** http://groups.csail.mit.edu/graphics/ilfr
*** http://dx.doi.org/10.1145/2185520.2185547
***********************************************************************

System requirements
===================

- Microsoft Windows XP, Vista, or 7. Developed and tested only on 
Windows 7 x64. 

- An OpenGL 2.0+ capable GPU. Required for the user interface regardless
of whether or not GPU reconstruction is used.

- For filtering large sample sets, several gigabytes of memory and a 
64-bit operating system. 

- For GPU reconstruction: 

	* NVIDIA CUDA-compatible GPU with compute capability 2.0 and at least 
	512 megabytes of RAM. GeForce GTX 480 or above is recommended. The 
	implementation is optimized for the Fermi architecture (e.g. GTX 480, 
	580).
	
	* NVIDIA CUDA 5.0 or later. Built and tested on 5.0 Release 
	Candidate. (See http://developer.nvidia.com/cuda-toolkit-archive)

	* Microsoft Visual Studio 2010 with Service Pack 1. Required even if you 
	do not plan to build the source code, as CUDA compilation, which happens 
	at runtime, requires it. 

- This software runs and compiles only on Windows and Visual Studio 2010 
Service Pack 1. We welcome contributions of ports to other versions of 
Visual Studio and other OSs. 



Instructions
============

General use
-----------

Launching reconstruction_app.exe will start the viewer application,
which by default loads a sample buffer from the data/ directory. The
default view (F1) shows the input samples with box filtering, which
results in a noisy image.

Pressing F2 (or clicking the corresponding button in the GUI) will run
our GPU reconstruction algorithm using diffuse settings and display
the result, provided you have a CUDA device with compute capability of
2.0 or over installed. F3 performs reconstruction using the CPU version.

F4 and F5 run the glossy code path (GPU/CPU). See notes below for details. 



Sample Buffer Format
--------------------

Sample buffers are stored in two files: the main file that contains
the sample data itself, and a second header file, whose name must
match the main file. For example,

	samplebuffer.txt
	samplebuffer.txt.header

form a valid pair.

The header specifies all kinds of useful information. It looks like this:

	Version 2.2
	Width 362
	Height 620
	Samples per pixel 8
	CoC coefficients (coc radius = C0/w+C1): -0.000000,0.000000
	Pixel-to-camera matrix: 0.004132,0.000000,0.000000,-0.750000,0.000000,-0.004132,0.000000,1.652893,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,-99.999001,100.000000
	Encoding = binary
	x,y,w,u,v,t,r,g,b,pri_mv(3d),pri_normal(3d),albedo(3d),sec_origin(3d),sec_hitpoint(3d),sec_mv(3d),sec_normal(3d),direct(3d),sec_albedo(3d),sec_direct(3d)

Width and Height specify the dimensions of the image, in pixels. Samples 
per pixel says how many samples to expect. The CoC coefficients give 
formulas for computing the slope dx/du and dy/dv given the camera space 
depth (z) for a sample (see below). Pixel-to-camera matrix is a 4x4 
projection matrix that maps pixel coordinates to points on the camera's 
focal plane (in camera space coordinates) as seen from the center of the 
lens. This information is only utilized in the DoF/motion reconstruction 
pipeline (see below, Figure 11 and Section 3.4). The last row serves as 
a reminder on how to interpret the numbers in the actual sample file. 

The actual sample data file can be either text or binary. We recommend 
generating your sample buffers in text format and using the 
functionality in UVTSampleBuffer to convert it to binary; this can be 
done by loading in the text version and serializing back to disk with 
the binary flag turned on. 

In the main sample file, each line describes one sample as a sequence 

x and y are the sample's pixel coordinates, including fractional 
subpixel offsets. 

w is the angular bandwidth estimate of the reflectance function at the 
secondary hit (see Section 2.5). IMPORTANT NOTE: The unintuitive naming 
(w) in and the location as the third parameter is due to historical 
reasons. 

u and v are the lens coordinates at which the sample was taken, in the 
range [-1,1]. Not used except in DoF/motion reconstruction. 

t is the time coordinate in the range [0,1] denoting the instant the 
sample was taken. Not used except in DoF/motion reconstruction. 

r,g,b is the sample's radiance, in linear RGB. Corresponds to $L$ in 
Section 2.1. 

pri_mv is the 3D camera-space motion vector for the surface hit by the 
primary ray. CURRENTLY UNUSED. 

pri_normal is the 3D camera-space normal for the primary hit.

albedo is the albedo for the primary hit. This variable is only used for 
diffuse reconstruction where we need to know the primary hit's 
reflectance in order to generate the final shaded value (glossy rays are 
handled separately as explained above). 

sec_origin is the origin of the secondary ray, and corresponds to 
$\mathbf{o}$ in Section 2.1. 

sec_hitpoint is the hit point of the secondary ray, and corresponds to 
$\mathbf{h}$ in Section 2.1.

sec_mv is the 3D camera-space motion vector for the secondary hit. 
CURRENTLY UNUSED except in the DoF/motion reconstruction, which abuses 
this field for motion of the primary hit. 

sec_normal is the 3D camera space normal of the secondary hit point.

direct is the RGB radiance of the direct lighting estimate at the 
primary hit. CURRENTLY UNUSED.

sec_albedo and sec_direct are the path throughput and direct radiance at 
the secondary hit, respectively. These are used only in the experimental 
recursive filtering (Sec. 4) in the CUDA reconstruction pipeline. 
Recursive filtering can be enabled by setting "gatherPasses" to a 
positive number in ReconstructionIndirectCUDA.cpp. 

The "CoC coefficients" C0,C1 are constants that are used for computing
the circle of confusion for a given depth, assuming a thin lens
model. The CoC corresponds directly to the slopes dx/du and dy/dv for
a given depth w. This is how to compute C0 and C1, illustrated using
PBRT's perspective camera API:

	float f = 1.f / ( 1 + 1.f / pCamera->getFocalDistance() );
	float sensorSize = 2 * tan( 0.5f * pCamera->getFoVRadians() );
	float cocCoeff1 = pCamera->getLensRadius() * f / ( pCamera->getFocalDistance() - f );
	cocCoeff1 *= min( camera->film->xResolution, camera->film->yResolution ) / sensorSize;
	float cocCoeff0 = -cocCoeff1 * pCamera->getFocalDistance();

If you use another model, you must derive the constants C0 and C1 yourself.
	

CUDA
----

The precompiled 64-bit executable is built with CUDA support, but
it is by default disabled in the code to enable building without the
CUDA SDK. To enable CUDA support, find the line

	#define FW_USE_CUDA 0
  
in src/framework/base/DLLImports.hpp and change it to

	#define FW_USE_CUDA 1
  
Provided you have a working install of CUDA Toolkit, this will enable 
the GPU code path. The first time it is run, the application will 
compile and cache the .cu file containing the reconstruction kernels. 
This may take a few seconds. 

If you get an error during initialization, the most probable
explanation is that the application is unable to launch nvcc.exe
contained in the CUDA Toolkit. In this case, you should:

   - Set CUDA_BIN_PATH to point to the CUDA Toolkit "bin" directory, e.g.
     "set CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\bin".

   - Set CUDA_INC_PATH to point to the CUDA Toolkit "include" directory, e.g.
     "set CUDA_INC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include".

   - Run vcvars32.bat to setup Visual Studio paths, e.g.
     "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\vcvars32.bat".


Release notes
=============

The GPU version is recommended for its much higher performance, while 
the CPU may provide better understanding of the code. The results are 
mostly identical. 

The CPU reconstruction code supports an optional scissor rectangle to 
limit computation and thus aid debugging. It can be set from App.cpp 
when calling the CPU reconstruction, e.g., tg.reconstructAO(...). 

The CPU code path supports a motion bounding volume hierarchy 
(motion-BVH) by default. We chose to default to this path as it 
simplifies the dof/motion reconstruction comparisons as it results in 
less code, but the generality adds approx. 10% of runtime compared to 
the runs in the paper. 

For glossy reconstruction, our prototype implements the generation of 
reconstruction rays and their BRDF weights by reading them in from a 
file ("ray dumps") generated using a modified version of PBRT, rather 
than replicating the sampling code. Pressing F4/F5 pops up a dialog 
asking for the ray buffer to use. 

As the ray buffers are too large for inclusion in the distribution 
package, we have provided a batch file in the "data/glossy-src" 
directory that generates sample buffers, the corresponding ray dumps, 
and ground truth images. Running the batch file will take a while and 
produce a large ray dump file of around 20 gigabytes. You can replace 
our reconstruction rays with yours from another source if desired, and 
the ray dump file provides a tight interface for doing so. The GPU 
version of the glossy ray dump renderer supports large files by 
streaming through them. The CPU version only supports smaller in-memory 
buffers. 

The application uses CUDA for gamma correction. If CUDA is not 
installed, the USE_CUDA define should be disabled. 

