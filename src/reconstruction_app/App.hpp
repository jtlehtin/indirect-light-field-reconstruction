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
#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"
#include "3d/CameraControls.hpp"
#include "reconstruction/Reconstruction.hpp"

namespace FW
{
//------------------------------------------------------------------------

class App : public Window::Listener, public CommonControls::StateObject
{
private:
    enum Action
    {
        Action_None,
		Action_LoadSampleBuffer,
		Action_SaveSampleBuffer,
		Action_ClearImages,
    };

	enum ReconstructionMode
	{
		RECONSTRUCT_INDIRECT = 0,
		RECONSTRUCT_AO = 1,
	};

	enum Visualization
	{
		VIZ_INPUT = 0,

		VIZ_RECONSTRUCTION_INDIRECT_CUDA,
		VIZ_RECONSTRUCTION_INDIRECT,
		VIZ_RECONSTRUCTION_GLOSSY_CUDA,
		VIZ_RECONSTRUCTION_GLOSSY,
		VIZ_RECONSTRUCTION_DOF_MOTION,
		VIZ_RECONSTRUCTION_RPF,				
		VIZ_RECONSTRUCTION_ATROUS,				

		VIZ_DEBUG,					// temp surface for receiving debug info
		VIZ_MAX,
	};

	enum Channel
	{
		CH_COLOR = 0,
		CH_DIRECT = 1,
		CH_INDIRECT = 2,
		CH_BOTH = 3,
		CH_NORMAL = 4,
		CH_ALBEDO = 5,
		CH_AO = 6,
		CH_MV = 7,
		CH_SALBEDO = 8, // secondary hit albedo
		CH_SDIRECT = 9, // secondary hit direct
		CH_BANDWIDTH = 10,
		CH_NORMAL_SMOOTH = 11,
	};

public:
                    App             	(void);
    virtual         ~App            	(void);

    virtual bool    handleEvent     	(const Window::Event& ev);
    virtual void    readState       	(StateDump& d);
    virtual void    writeState      	(StateDump& d) const;

private:
    void            render				(GLContext* gl);
    void            importSampleBuffer	(const String& fileName);

	void			getChannel			(Image& img, Channel ch) const;

	void			blitToWindow		(GLContext* gl, Image& img);
	void			adjustGamma			(Image& image) const;
    void            firstTimeInit   	(void);

private:
                    App             	(const App&); // forbidden
    App&            operator=       	(const App&); // forbidden

private:
    Window          	m_window;
    CommonControls  	m_commonCtrl;
	CameraControls		m_camera;

    Action          	m_action;

	UVTSampleBuffer*	m_samples;
	Image*				m_images[VIZ_MAX];
	
	String				m_fileName;
	bool				m_haveSampleBuffer;
	U32					m_vizDone;

	bool				m_flipY;
	bool				m_exportScreenshot;
	float				m_gamma;
	float				m_aoLength;
	S32					m_numReconstructionRays;

	S32					m_showImage;
	S32					m_showChannel;
	S32					m_reconstructionMode;
};

//------------------------------------------------------------------------
}
