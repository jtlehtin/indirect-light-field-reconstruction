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
