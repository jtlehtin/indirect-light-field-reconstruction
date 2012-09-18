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

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4127)

#include "App.hpp"
#include "base/Main.hpp"
#include "base/Timer.hpp"
#include "gpu/GLContext.hpp"
#include "io/Stream.hpp"
#include "io/StateDump.hpp"
#include "io/ImageLodePngIO.hpp"
#include "gpu/CudaCompiler.hpp"

#include <stdio.h>
#include <conio.h>

using namespace FW;

//------------------------------------------------------------------------

App::App(void)
:   m_commonCtrl    					(CommonControls::Feature_Default & ~(CommonControls::Feature_RepaintOnF5 | CommonControls::Feature_ShowFPSOnF9 | CommonControls::Feature_HideControlsOnF10 | CommonControls::Feature_FullScreenOnF11)),
	m_camera		  					(&m_commonCtrl, 0),
    m_action        					(Action_None),
	m_samples							(NULL),
	m_haveSampleBuffer					(false),
	m_vizDone							(0),
	m_flipY								(true),
	m_exportScreenshot					(false),
	m_gamma								(1.f),
	m_aoLength							(3.5f),
	m_numReconstructionRays				(256),
	m_showImage							(0),
	m_showChannel						(CH_INDIRECT),
	m_reconstructionMode				(RECONSTRUCT_INDIRECT)
{
	for(int i=0;i<VIZ_MAX;i++)
		m_images[i] = NULL;

    m_commonCtrl.showFPS(true);
    m_commonCtrl.addStateObject(this);
	m_camera.setKeepAligned(true);

    m_commonCtrl.addButton((S32*)&m_action, Action_LoadSampleBuffer,    FW_KEY_L,       "Load sample buffer... [L]");
    m_commonCtrl.addButton((S32*)&m_action, Action_SaveSampleBuffer,    FW_KEY_S,       "Save sample buffer... [S]");
	m_commonCtrl.addToggle(&m_flipY,									FW_KEY_Y,		"Flip Y [Y]");
	m_commonCtrl.addToggle(&m_exportScreenshot,							FW_KEY_P,		"Output screenshot [P]");
	m_commonCtrl.addButton((S32*)&m_action, Action_ClearImages,			FW_KEY_DELETE,	"Force recalculate [DELETE]");
	m_window.addListener(&m_camera);

    m_commonCtrl.addSeparator();

	m_commonCtrl.addToggle(&m_reconstructionMode, RECONSTRUCT_INDIRECT,	FW_KEY_NONE,  	"Reconstruction mode: indirect");
	m_commonCtrl.addToggle(&m_reconstructionMode, RECONSTRUCT_AO	,	FW_KEY_NONE,  	"Reconstruction mode: AO");

    m_commonCtrl.addSeparator();

	m_commonCtrl.addToggle(&m_showImage, VIZ_INPUT						, FW_KEY_F1, 	"Show input channel [F1]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_INDIRECT_CUDA,FW_KEY_F2,  	"Lehtinen et al. 2012 Indirect/AO with CUDA [F2]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_INDIRECT	, FW_KEY_F3,  	"Lehtinen et al. 2012 Indirect/AO with CPU  [F3]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_GLOSSY_CUDA,  FW_KEY_F4,  	"Lehtinen et al. 2012 Glossy with CUDA [F4]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_GLOSSY	    , FW_KEY_F5,  	"Lehtinen et al. 2012 Glossy with CPU  [F5]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_DOF_MOTION  , FW_KEY_F6,  	"Lehtinen et al. 2012 Dof + Motion with CPU [F6]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_RPF			, FW_KEY_F7,  	"RPF filter    [F7]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_ATROUS		, FW_KEY_F8,  	"ATrous filter [F8]");
//	m_commonCtrl.addSeparator();
	m_commonCtrl.addToggle(&m_showImage, VIZ_DEBUG						, FW_KEY_F9, 	"Show debug surface [F9]");

    m_commonCtrl.addSeparator();

//	m_commonCtrl.setControlVisibility(false);
	m_commonCtrl.addToggle(&m_showChannel, CH_DIRECT	, FW_KEY_D,  	"Input channel: DIRECT [D]");
	m_commonCtrl.addToggle(&m_showChannel, CH_INDIRECT	, FW_KEY_I,  	"Input channel: INDIRECT [I]");
	m_commonCtrl.addToggle(&m_showChannel, CH_BOTH		, FW_KEY_B,  	"Input channel: BOTH [B]");
	m_commonCtrl.addToggle(&m_showChannel, CH_NORMAL	, FW_KEY_N,  	"Input channel: NORMAL [N]");
	m_commonCtrl.addToggle(&m_showChannel, CH_NORMAL_SMOOTH, FW_KEY_H, 	"Input channel: SMOOTH NORMAL [H]");
	m_commonCtrl.addToggle(&m_showChannel, CH_ALBEDO	, FW_KEY_A,  	"Input channel: ALBEDO [A]");
	m_commonCtrl.addToggle(&m_showChannel, CH_AO		, FW_KEY_O,  	"Input channel: AO [O]");
	m_commonCtrl.addToggle(&m_showChannel, CH_MV		, FW_KEY_M,  	"Input channel: MV [M]");
	m_commonCtrl.addToggle(&m_showChannel, CH_SALBEDO   , FW_KEY_Z,  	"Input channel: SECONDARY ALBEDO [Z]");
	m_commonCtrl.addToggle(&m_showChannel, CH_SDIRECT   , FW_KEY_X,  	"Input channel: SECONDARY DIRECT [X]");
	m_commonCtrl.addToggle(&m_showChannel, CH_BANDWIDTH , FW_KEY_C,  	"Input channel: BANDWIDTH [C]");
	m_commonCtrl.addToggle(&m_showChannel, CH_COLOR     , FW_KEY_V,  	"Input channel: COLOR [V]");
//	m_commonCtrl.setControlVisibility(true);

    m_window.setTitle("Indirect lightfield reconstruction");
	m_window.setSize(Vec2i(640,480));
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);

	// sliders for camera parameters
	m_commonCtrl.beginSliderStack();
	m_commonCtrl.addSlider(&m_gamma,    1.f,2.5f,false, FW_KEY_NONE,FW_KEY_NONE, "Gamma %.2f", 0.1f);
	m_commonCtrl.addSlider(&m_aoLength, 1.f,1000.f,true, FW_KEY_NONE,FW_KEY_NONE, "AO ray length %.1f", 0.1f);
	m_commonCtrl.addSlider(&m_numReconstructionRays, 32,1024,true, FW_KEY_NONE,FW_KEY_NONE, "#reconstruction rays %d", 1);
	m_commonCtrl.endSliderStack();

	// foo
	{
		CudaCompiler compiler;
		compiler.setSourceFile("src/reconstruction_lib/reconstruction/ReconstructionIndirectCudaKernels.cu");
        compiler.addOptions("-use_fast_math");
		compiler.include("src/framework");
		compiler.define("SM_ARCH", sprintf("%d", CudaModule::getComputeCapability()));
		compiler.compile();
	}

	// load state
    m_commonCtrl.loadState(m_commonCtrl.getStateFileName(1));
}

//------------------------------------------------------------------------

App::~App(void)
{
	delete m_samples;
	for(int i=0;i<VIZ_MAX;i++)
		delete m_images[i];
}

//------------------------------------------------------------------------

bool App::handleEvent(const Window::Event& ev)
{
	if (ev.type == Window::EventType_Close)
    {
        m_window.showModalMessage("Exiting...");
        delete this;
        return true;
    }

    Action action = m_action;
    m_action = Action_None;
    String name;
    Mat4f mat;

    switch (action)
    {
    case Action_None:
        break;

	case Action_LoadSampleBuffer:
        name = m_window.showFileLoadDialog("Load sample buffer");
        if (name.getLength())
            importSampleBuffer(name);
        break;

	case Action_SaveSampleBuffer:
        name = m_window.showFileSaveDialog("Save sample buffer");
        if (name.getLength())
		{
			m_samples->serialize(name.getPtr(), true, true);
			m_fileName = name;
		}
        break;

	case Action_ClearImages:
		m_vizDone = 0;
		break;

    default:
        FW_ASSERT(false);
        break;
    }

    m_window.setVisible(true);

    if (ev.type == Window::EventType_Paint)
        render(m_window.getGL());
    m_window.repaint();
    return false;
}

//------------------------------------------------------------------------

void App::readState(StateDump& d)
{
	printf("readState\n");

    d.pushOwner("App");
    String fileName;
	d.get(fileName, "m_fileName");
    d.get((S32&)m_flipY, "m_flipY");
    d.get((F32&)m_gamma, "m_gamma");
    d.popOwner();

	if(fileName.getLength())
		importSampleBuffer(fileName);
}

//------------------------------------------------------------------------

void App::writeState(StateDump& d) const
{
	printf("writeState\n");

    d.pushOwner("App");
	d.set(m_fileName, "m_fileName");
    d.set((S32&)m_flipY, "m_flipY");
    d.set((F32&)m_gamma, "m_gamma");
    d.popOwner();
}

//------------------------------------------------------------------------

void App::firstTimeInit(void)
{
}

//------------------------------------------------------------------------

void FW::init(void)
{
    CudaCompiler::setFrameworkPath("src/framework");
	new App;
}

//------------------------------------------------------------------------

void filterNormals(UVTSampleBuffer& sbuf)
{
	// Basically this is needed because our PBRT exporter outputs the *geometric* normal, and image filters such as A-Trous and RPF want a smooth shading normal.
	// Ideally the smooth normal would be stored in the buffer, and this approximate operation wouldn't be needed. So, basically this is a hack due to our content pipeline.

	// Does the smooth primary normals channel exist?
	int CID_PRI_NORMAL_SMOOTH = sbuf.getChannelID(CID_PRI_NORMAL_SMOOTH_NAME);
	if(CID_PRI_NORMAL_SMOOTH!=-1)
		return;

	FW::printf("Smoothing normal channel...");

	CID_PRI_NORMAL_SMOOTH = sbuf.reserveChannel<Vec3f>(CID_PRI_NORMAL_SMOOTH_NAME);

	const int CID_PRI_NORMAL = sbuf.getChannelID(CID_PRI_NORMAL_NAME);
	const int w = sbuf.getWidth();
	const int h = sbuf.getHeight();
	const int fw = 5;
	const int fr = fw/2;

	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	for(int i=0;i<sbuf.getNumSamples(x,y);i++)
	{
		const Vec2f& xyi = sbuf.getSampleXY(x,y,i);
		const Vec3f& ni  = sbuf.getSampleExtra<Vec3f>(CID_PRI_NORMAL, x,y,i);
		Vec4f normal(0);
		for(int dx=-fr;dx<=fr;dx++)
		for(int dy=-fr;dy<=fr;dy++)
		{
			const int sx = x+dx;
			const int sy = y+dy;
			if(sx<0 || sy<0 || sx>=w || sy>=h)
				continue;

			for(int j=0;j<sbuf.getNumSamples(sx,sy);j++)
			{
				const Vec2f& xyj = sbuf.getSampleXY(sx,sy,j);
				const Vec3f& nj  = sbuf.getSampleExtra<Vec3f>(CID_PRI_NORMAL, sx,sy,j);

				// spatial (screen)
				const float xydist2  = (xyj - xyi).lenSqr();
				const float xyradius = fw/2.f;
				const float xystddev = xyradius / 2.f;								// 2 stddevs (98%) at the filter border
				float d = xydist2/(2*sqr(xystddev));

				// normal
				const float ndist2  = (nj - ni).lenSqr();
				const float nstddev = 0.5f;
				d += ndist2/(2*sqr(nstddev));

				// combined
				const float weight = FW::exp( -d );
				normal += weight*Vec4f(nj,1);
			}
		}
		normal *= rcp(normal.w);
		sbuf.setSampleExtra<Vec3f>(CID_PRI_NORMAL_SMOOTH, x,y,i, normal.getXYZ());
	}

	FW::printf("done\n");
}

//------------------------------------------------------------------------

void App::importSampleBuffer (const String& fileName)
{
	m_fileName         = fileName;
	m_haveSampleBuffer = true;
	m_vizDone		   = 0;
	m_gamma			   = 1.f;

	// Import sample buffer.

	delete m_samples;
	FILE* fp = fopen(fileName.getPtr(), "rb");	// try to open
	if(fp)
	{
		fclose(fp);
		m_samples = new UVTSampleBuffer(fileName.getPtr());
		if(m_samples->isIrregular())
			printf("IRREGULAR sample buffer loaded\n");
	}
	else
	{
		delete m_samples;
		m_samples = new UVTSampleBuffer(m_window.getSize().x,m_window.getSize().y,1);
		printf("ERROR: File %s not found\n", fileName.getPtr());
	}

	// Resize window and images.

	const Vec2i windowSize( m_samples->getWidth(),m_samples->getHeight() );
	m_window.setSize( windowSize );

	for(int i=0;i<VIZ_MAX;i++)
	{
		delete m_images[i];
		m_images[i] = new Image(windowSize, ImageFormat::RGBA_Vec4f);
		m_images[i]->clear();
	}

	// compute input image by bucketing the input samples (box filter)
	getChannel(*m_images[VIZ_INPUT], CH_COLOR);
}

//------------------------------------------------------------------------

void App::getChannel(Image& img, Channel ch) const
{
	if(ch != CH_COLOR && m_samples->getVersion() < 2.f)
		return;

	const int CID_DIRECT       = m_samples->getChannelID(CID_DIRECT_NAME);
	const int CID_PRI_NORMAL   = m_samples->getChannelID(CID_PRI_NORMAL_NAME);
	const int CID_SEC_ORIGIN   = m_samples->getChannelID(CID_SEC_ORIGIN_NAME);
	const int CID_SEC_HITPOINT = m_samples->getChannelID(CID_SEC_HITPOINT_NAME);
	const int CID_SEC_MV	   = m_samples->getChannelID(CID_SEC_MV_NAME);
	const int CID_ALBEDO       = m_samples->getChannelID(CID_ALBEDO_NAME);
	const int CID_SEC_ALBEDO   = m_samples->getChannelID(CID_SEC_ALBEDO_NAME);
	const int CID_SEC_DIRECT   = m_samples->getChannelID(CID_SEC_DIRECT_NAME);

	if(ch==CH_NORMAL_SMOOTH)
		filterNormals(*m_samples);

	const int CID_PRI_NORMAL_SMOOTH = m_samples->getChannelID(CID_PRI_NORMAL_SMOOTH_NAME);

	for(int y=0;y<m_samples->getHeight();y++)
	for(int x=0;x<m_samples->getWidth();x++)
	{
		Vec4f pixelColor(0);

		for(int i=0;i<m_samples->getNumSamples(x,y);i++)
		if(m_samples->getSampleExtra<Vec3f>(CID_SEC_ORIGIN, x,y,i).max() < 1e10f)		// is valid?
		{
			Vec2f p = m_samples->getSampleXY(x,y,i);
			Vec2i pi((int)floor(p.x),(int)floor(p.y));
			FW_ASSERT(pi.x>=0 || pi.y>=0 || pi.x<img.getSize().x || pi.y<img.getSize().y);

			switch(ch)
			{
			case CH_COLOR:		pixelColor += m_samples->getSampleColor(x,y,i); break;
			case CH_DIRECT:		pixelColor += Vec4f(m_samples->getSampleExtra<Vec3f>(CID_DIRECT, x,y,i),1); break;
			case CH_NORMAL:		pixelColor += Vec4f(m_samples->getSampleExtra<Vec3f>(CID_PRI_NORMAL, x,y,i)/2+0.5f,1); break;
			case CH_NORMAL_SMOOTH: pixelColor += Vec4f(m_samples->getSampleExtra<Vec3f>(CID_PRI_NORMAL_SMOOTH, x,y,i)/2+0.5f,1); break;
			case CH_ALBEDO:		pixelColor += Vec4f(m_samples->getSampleExtra<Vec3f>(CID_ALBEDO, x,y,i),1); break;
			case CH_SALBEDO:	pixelColor += Vec4f(m_samples->getSampleExtra<Vec3f>(CID_SEC_ALBEDO, x,y,i),1); break;
			case CH_SDIRECT:	pixelColor += Vec4f(m_samples->getSampleExtra<Vec3f>(CID_SEC_DIRECT, x,y,i),1); break;
			case CH_MV:			pixelColor += Vec4f(m_samples->getSampleExtra<Vec3f>(CID_SEC_MV, x,y,i)/2.0f+0.5f,1); break;
			case CH_AO:			pixelColor += (m_samples->getSampleExtra<Vec3f>(CID_SEC_HITPOINT, x,y,i)-m_samples->getSampleExtra<Vec3f>(CID_SEC_ORIGIN, x,y,i)).length() < m_aoLength ? Vec4f(0,0,0,1) : Vec4f(1,1,1,1); break;
			case CH_BANDWIDTH:  pixelColor += Vec4f(m_samples->getSampleW(x,y,i) * .01f, 1.f); break;
			case CH_INDIRECT:
				{
					const Vec3f& incident = m_samples->getSampleColor(x,y,i).getXYZ();
					const Vec3f& albedo   = m_samples->getSampleExtra<Vec3f>(CID_ALBEDO, x,y,i);
					pixelColor += Vec4f(incident*albedo,1);
					break;
				}
			case CH_BOTH:
				{
					const Vec3f& incident = m_samples->getSampleColor(x,y,i).getXYZ();
					const Vec3f& albedo   = m_samples->getSampleExtra<Vec3f>(CID_ALBEDO, x,y,i);
					const Vec3f& direct   = m_samples->getSampleExtra<Vec3f>(CID_DIRECT, x,y,i);
					pixelColor += Vec4f(direct + incident*albedo,1);				// ASSUMES: sample density according to BDRF
					break;
				}
			default:
				fail("App::getChannel, unsupported channel requested");
			}
		}

		if(pixelColor.w)	img.setVec4f(Vec2i(x,y), pixelColor*rcp(pixelColor.w));
		else				img.setVec4f(Vec2i(x,y), Vec4f(0,1,0,1));					// NO SUPPORT = GREEN
	}
}

//------------------------------------------------------------------------

void App::render (GLContext* gl)
{
	// no sample buffer?
    if (!m_samples)
    {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }

	static int messageTimer = 0;
	static String messageString;

	// Copy requested buffer to image.

	const Image* img = NULL;
	const Visualization viz = (Visualization)m_showImage;
	const float aoLength = (m_reconstructionMode==RECONSTRUCT_AO ? m_aoLength : 0);

	switch(viz)
	{
	// nop
	case VIZ_INPUT:
		{
			getChannel(*m_images[viz], (Channel)m_showChannel);
			img = m_images[viz];
			break;
		}
	// indirect/AO
	case VIZ_RECONSTRUCTION_INDIRECT_CUDA:
	case VIZ_RECONSTRUCTION_INDIRECT:
		{
			Reconstruction tg;
			if(m_haveSampleBuffer && !(m_vizDone&(1<<viz)))
			{
				m_vizDone |= (1<<viz);
				m_vizDone |= (1<<VIZ_DEBUG);						// HACK
				getChannel(*m_images[viz],CH_INDIRECT);				// for active window

				if(viz==VIZ_RECONSTRUCTION_INDIRECT_CUDA && !CudaModule::isAvailable())
				{
					messageTimer  = 75;
					messageString = String("CUDA not available");
					printf("%s\n", messageString.getPtr());
				}
				else
				{
					if(aoLength==0)
					{
						if(viz==VIZ_RECONSTRUCTION_INDIRECT_CUDA)	tg.reconstructIndirectCuda(*m_samples,m_numReconstructionRays,*m_images[viz]);
						else										tg.reconstructIndirect    (*m_samples,m_numReconstructionRays,*m_images[viz],m_images[VIZ_DEBUG]);
					}
					else
					{
						if(viz==VIZ_RECONSTRUCTION_INDIRECT_CUDA)	tg.reconstructAOCuda(*m_samples,m_numReconstructionRays,aoLength,*m_images[viz]);
						else										tg.reconstructAO    (*m_samples,m_numReconstructionRays,aoLength,*m_images[viz],m_images[VIZ_DEBUG]);//, Vec4i(401,401,500,500));
					}
				}
			}
			img = m_images[viz];
			break;
		}
	// glossy
	case VIZ_RECONSTRUCTION_GLOSSY_CUDA:
	case VIZ_RECONSTRUCTION_GLOSSY:
		{
			Reconstruction tg;
			if(m_haveSampleBuffer && !(m_vizDone&(1<<viz)))
			{
				m_vizDone |= (1<<viz);
				m_vizDone |= (1<<VIZ_DEBUG);						// HACK
				getChannel(*m_images[viz],CH_INDIRECT);				// for active window

				if(viz==VIZ_RECONSTRUCTION_GLOSSY_CUDA && !CudaModule::isAvailable())
				{
					messageTimer  = 75;
					messageString = String("CUDA not available");
					printf("%s\n", messageString.getPtr());
				}
				else
				{
					// ask for a ray dump
					String rayDumpFileName = m_window.showFileLoadDialog("Load ray dump");
					if (rayDumpFileName.getLength())
					{
						if(viz==VIZ_RECONSTRUCTION_GLOSSY_CUDA)	tg.reconstructGlossyCuda(*m_samples,rayDumpFileName,*m_images[viz]);
						else									tg.reconstructGlossy    (*m_samples,rayDumpFileName,*m_images[viz],m_images[VIZ_DEBUG]);
					}
					else
					{
						messageTimer  = 50;
						messageString = String("Ray dump not loaded");
						printf("Ray dump not loaded\n");
					}
				}
			}
			img = m_images[viz];
			break;
		}
	// dof+motion
	case VIZ_RECONSTRUCTION_DOF_MOTION:
		{
			Reconstruction tg;
			if(m_haveSampleBuffer && !(m_vizDone&(1<<viz)))
			{
				m_vizDone |= (1<<viz);
				m_vizDone |= (1<<VIZ_DEBUG);						// HACK
				getChannel(*m_images[viz],CH_INDIRECT);				// for active window
				tg.reconstructDofMotion(*m_samples,m_numReconstructionRays,*m_images[viz],m_images[VIZ_DEBUG]);
//				tg.reconstructDofMotion(*m_samples,m_numReconstructionRays,*m_images[viz],m_images[VIZ_DEBUG], Vec4i(701,101,800,400));	// reconstruct a partial image
			}
			img = m_images[viz];
			break;
		}
	// RPF
	case VIZ_RECONSTRUCTION_RPF:
		{
			Reconstruction tg;
			if(m_haveSampleBuffer && !(m_vizDone&(1<<viz)))
			{
				m_vizDone |= (1<<viz);
				m_vizDone |= (1<<VIZ_DEBUG);						// HACK
				getChannel(*m_images[viz],CH_INDIRECT);				// for active window
				filterNormals(*m_samples);
				tg.reconstructRPF(*m_samples,*m_images[viz],m_images[VIZ_DEBUG],aoLength);			
			}
			img = m_images[viz];
			break;
		}
	// ATrous
	case VIZ_RECONSTRUCTION_ATROUS:
		{
			Reconstruction tg;
			if(m_haveSampleBuffer && !(m_vizDone&(1<<viz)))
			{
				m_vizDone |= (1<<viz);
				m_vizDone |= (1<<VIZ_DEBUG);						// HACK
				getChannel(*m_images[viz],CH_INDIRECT);				// for active window
				filterNormals(*m_samples);
				tg.reconstructATrous(*m_samples,*m_images[viz],m_images[VIZ_DEBUG],aoLength);			
			}
			img = m_images[viz];
			break;
		}
	// default
	default: fail("App::render");
	}

	Image gc(*img);
	adjustGamma(gc);

	// screenshot
	if(m_exportScreenshot)
	{
		m_exportScreenshot = false;

		SYSTEMTIME st;
		FILETIME ft;
		GetSystemTime(&st);
		SystemTimeToFileTime(&st, &ft);
		U64 stamp = *(const U64*)&ft;

		String name("screenshot_x_");
		for (int i = 60; i >= 0; i -= 4)
			name += (char)('a' + ((stamp >> i) & 15));
		name += ".png";

		exportImage(name, &gc);
		printf("Exported screenshot\n");
	}

	// show the result
	blitToWindow(gl, gc);

	// show message
	if(messageTimer>0)
	{
		messageTimer--;
		gl->drawModalMessage(messageString.getPtr());
	}
}

//------------------------------------------------------------------------

void App::blitToWindow(GLContext* gl, Image& img)
{
	m_window.setSize( img.getSize() );

	Mat4f oldXform = gl->setVGXform(Mat4f());
	glPushAttrib(GL_ENABLE_BIT);
	glDisable(GL_DEPTH_TEST);
	gl->drawImage(img, Vec2f(0.0f), 0.5f, m_flipY);
	gl->setVGXform(oldXform);
	glPopAttrib();
}

//------------------------------------------------------------------------

void App::adjustGamma(Image& img) const
{
	if(CudaModule::isAvailable())
	{
		CudaModule* module = FW_COMPILE_INLINE_CUDA
		(
			__global__ void adjustGamma(Vec4f* image, int numPixels, float gamma)
			{
				int index = globalThreadIdx;
				if (index >= numPixels)
					return;

				Vec4f& c = image[index];
				c.x = powf(c.x, 1/gamma);
				c.y = powf(c.y, 1/gamma);
				c.z = powf(c.z, 1/gamma);
			}
		);

		int numPixels = img.getSize().x * img.getSize().y;
		module->getKernel("adjustGamma").setParams(img, numPixels, m_gamma).launch(numPixels).sync(false);
	}
	else
	{
		for(int y=0;y<img.getSize().y;y++)
		for(int x=0;x<img.getSize().x;x++)
		{
			Vec4f c = img.getVec4f(Vec2i(x,y));
			c.x = pow(c.x, 1/m_gamma);
			c.y = pow(c.y, 1/m_gamma);
			c.z = pow(c.z, 1/m_gamma);
			img.setVec4f(Vec2i(x,y), c);
		}
	}
}

