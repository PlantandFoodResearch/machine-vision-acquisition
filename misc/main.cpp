#include <algorithm>
#include <iostream>
#include <set>
#include <thread>
#include "exampleHelper.h"
#include <mvIMPACT_CPP/mvIMPACT_acquire_helper.h>
#ifdef _WIN32
#   include <mvDisplay/Include/mvIMPACT_acquire_display.h>
using namespace mvIMPACT::acquire::display;
#   define USE_DISPLAY
#endif // #ifdef _WIN32
 
using namespace std;
using namespace mvIMPACT::acquire;
 
//-----------------------------------------------------------------------------
struct ThreadParameter
//-----------------------------------------------------------------------------
{
    Device* pDev_;
    unsigned int requestsCaptured_;
    Statistics statistics_;
#ifdef USE_DISPLAY
    ImageDisplayWindow displayWindow_;
#endif // #ifdef USE_DISPLAY
    explicit ThreadParameter( Device* pDev ) : pDev_( pDev ), requestsCaptured_( 0 ), statistics_( pDev )
#ifdef USE_DISPLAY
        // initialise display window
        // IMPORTANT: It's NOT safe to create multiple display windows in multiple threads!!!
        , displayWindow_( "mvIMPACT_acquire sample, Device " + pDev_->serial.read() )
#endif // #ifdef USE_DISPLAY
    {}
    ThreadParameter( const ThreadParameter& src ) = delete;
    ThreadParameter& operator=( const ThreadParameter& rhs ) = delete;
};
 
//-----------------------------------------------------------------------------
void myThreadCallback( shared_ptr<Request> pRequest, ThreadParameter& threadParameter )
//-----------------------------------------------------------------------------
{
    ++threadParameter.requestsCaptured_;
    // display some statistical information every 100th image
    if( threadParameter.requestsCaptured_ % 100 == 0 )
    {
        const Statistics& s = threadParameter.statistics_;
        cout << "Info from " << threadParameter.pDev_->serial.read()
             << ": " << s.framesPerSecond.name() << ": " << s.framesPerSecond.readS()
             << ", " << s.errorCount.name() << ": " << s.errorCount.readS()
             << ", " << s.captureTime_s.name() << ": " << s.captureTime_s.readS() << endl;
    }
    if( pRequest->isOK() )
    {
#ifdef USE_DISPLAY
        threadParameter.displayWindow_.GetImageDisplay().SetImage( pRequest );
        threadParameter.displayWindow_.GetImageDisplay().Update();
#else
        cout << "Image captured: " << pRequest->imageOffsetX.read() << "x" << pRequest->imageOffsetY.read() << "@" << pRequest->imageWidth.read() << "x" << pRequest->imageHeight.read() << endl;
#endif // #ifdef USE_DISPLAY
    }
    else
    {
        cout << "Error: " << pRequest->requestResult.readS() << endl;
    }
}
 
//-----------------------------------------------------------------------------
void modifyConnectorFromUserInput( Connector& connector )
//-----------------------------------------------------------------------------
{
    bool boRun = true;
    while( boRun )
    {
        cout << endl << "Current connector settings:" << endl
             << "  Camera output: " << connector.cameraOutputUsed.readS() << endl
             << "  Grabber input: " << connector.videoChannel.read() << "(" << connector.pinDescription.read() << ")" << endl;
        cout << "Press" << endl
             << "  'o' to select a different camera output" << endl
             << "  'i' to select a different grabber input" << endl
             << "  any other key when done" << endl;
        string cmd;
        cin >> cmd;
        // remove the '\n' from the stream
        cin.get();
        if( cmd == "o" )
        {
            cout << "Available camera outputs as defined by the camera description:" << endl;
            vector<pair<string, TCameraOutput> > v;
            // query the properties translation dictionary, which will contain every camera description
            // recognized by this device
            connector.cameraOutputUsed.getTranslationDict( v );
            for_each( v.begin(), v.end(), DisplayDictEntry<int>() );
            cout << "Enter the new (numerical) value: ";
            int newOutput;
            cin >> newOutput;
            try
            {
                connector.cameraOutputUsed.write( static_cast<TCameraOutput>( newOutput ) );
            }
            catch( const ImpactAcquireException& e )
            {
                cout << e.getErrorString() << endl;
            }
        }
        else if( cmd == "i" )
        {
            // The allowed range depends on the currently selected camera output as e.g. for
            // a RGB camera signal 3 video input are required on the device side, while a
            // composite signal just requires 1.
            cout << "Allowed grabber video inputs in the current camera output mode: " <<
                 connector.videoChannel.read( plMinValue ) << " - " <<
                 connector.videoChannel.read( plMaxValue ) << endl;
            cout << "Enter the new (numerical) value: ";
            int newVideoInput;
            cin >> newVideoInput;
            try
            {
                connector.videoChannel.write( newVideoInput );
            }
            catch( const ImpactAcquireException& e )
            {
                cout << e.getErrorString() << endl;
            }
        }
        else
        {
            boRun = false;
            continue;
        }
    }
}
 
//-----------------------------------------------------------------------------
void selectCameraDescriptionFromUserInput( CameraSettingsFrameGrabber& cs )
//-----------------------------------------------------------------------------
{
    // display the name of every camera description available for this device.
    // this might be less than the number of camera descriptions available on the system as e.g.
    // an analog frame grabber can't use descriptions for digital cameras
    vector<pair<string, int> > vAvailableDescriptions;
    cs.type.getTranslationDict( vAvailableDescriptions );
    cout << endl << "Available camera descriptions: " << vAvailableDescriptions.size() << endl
         << "----------------------------------" << endl;
    for_each( vAvailableDescriptions.begin(), vAvailableDescriptions.end(), DisplayDictEntry<int>() );
    cout << endl << "Please select a camera description to use for this sample (numerical value): ";
 
    int cameraDescriptionToUse;
    cin >> cameraDescriptionToUse;
    // remove the '\n' from the stream
    cin.get();
    // wrong input will raise an exception
    // The camera description contains all the parameters needed to capture the associated video
    // signal. Camera descriptions for a certain camera can be obtained from MATRIX VISION. Experienced
    // users can also create their own camera descriptions using wxPropView. The documentation will
    // explain how to create and configure camera description files
    cs.type.write( cameraDescriptionToUse );
}
 
//-----------------------------------------------------------------------------
bool isDeviceSupportedBySample( const Device* const pDev )
//-----------------------------------------------------------------------------
{
    return pDev->hasCapability( dcCameraDescriptionSupport );
}
 
//-----------------------------------------------------------------------------
int main( void )
//-----------------------------------------------------------------------------
{
    DeviceManager devMgr;
 
    cout << "This sample is meant for devices that support camera descriptions only. Other devices might be installed" << endl
         << "but won't be recognized by the application." << endl
         << endl;
 
    Device* pDev = getDeviceFromUserInput( devMgr, isDeviceSupportedBySample );
    if( pDev == nullptr )
    {
        cout << "Unable to continue! Press [ENTER] to end the application" << endl;
        cin.get();
        return 1;
    }
 
    cout << "Initialising device " << pDev->serial.read() << ". This might take some time..." << endl;
    try
    {
        pDev->open();
    }
    catch( const ImpactAcquireException& e )
    {
        // this e.g. might happen if the same device is already opened in another process...
        cout << "An error occurred while opening the device(error code: " << e.getErrorCode() << ")." << endl
             << "Press [ENTER] to end the application" << endl;
        cin.get();
        return 1;
    }
 
    // Assume that this is a frame grabber as currently only these devices support camera descriptions
    CameraSettingsFrameGrabber cs( pDev );
    if( !cs.type.isValid() )
    {
        cout << "Device " << pDev->serial.read() << " doesn't seem to support camera descriptions." << endl
             << "Press [ENTER] to end the application" << endl;
        cin.get();
        return 1;
    }
 
    Connector connector( pDev );
    CameraDescriptionManager cdm( pDev );
    bool boRun = true;
    while( boRun )
    {
        try
        {
            selectCameraDescriptionFromUserInput( cs );
            cdm.getStandardCameraDescriptionCount();
            modifyConnectorFromUserInput( connector );
 
            // start the execution of the 'live' thread.
            cout << endl << "Press [ENTER] to end the acquisition" << endl << endl;
            ThreadParameter threadParam( pDev );
            helper::RequestProvider requestProvider( pDev );
            requestProvider.acquisitionStart( myThreadCallback, std::ref( threadParam ) );
            cin.get();
            requestProvider.acquisitionStop();
            cout << endl << "Press 'q' followed by [ENTER] to end to application or any other key followed by [ENTER] to select a different camera description: ";
            string cmd;
            cin >> cmd;
            if( cmd == "q" )
            {
                boRun = false;
                continue;
            }
        }
        catch( const ImpactAcquireException& e )
        {
            cout << "Invalid selection (Error message: " << e.getErrorString() << ")" << endl
                 << "Please make a valid selection" << endl;
        }
    }
    return 0;
}