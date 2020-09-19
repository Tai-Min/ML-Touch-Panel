#include "networkthread.h"
#include "messages.h"
#include <regex>

InferenceEngine::Core NetworkThread::core = InferenceEngine::Core();

NetworkThread::~NetworkThread(){
    info("Calling threading clean up...\n");
    info(" - Destroying network thread object...\n");

    threadExitSignal.set_value();

    while(!isFinished()){}

    success(" - Network thread object destroyed.\n");
    success("Cleanup finished.\n");
}

void NetworkThread::run(){
    threadExitSignal = promise<void>();
    future<void> exitSignal = threadExitSignal.get_future();
    while(exitSignal.wait_for(chrono::microseconds(1)) == future_status::timeout){
        if(!runStep())
            return;
    }
}

string NetworkThread::getSimilarDevice(const string & dev, const DeviceList & devices) const {
    string chosenDevice = "";
    string strPattern = "^(" + dev + ")|(" + dev + "\\.[0-9]+)$";
    regex pattern(strPattern);
    smatch sm;
    for(const string & dev: devices){
        regex_match(dev, sm, pattern);
        if(sm.size()){
            chosenDevice = dev;
            break;
        }
    }
    return chosenDevice;
}

InferenceEngine::ExecutableNetwork NetworkThread::tryToLoadNetwork(const InferenceEngine::CNNNetwork & net, const string & device, bool isWarn, bool *ok) const {
    *ok = false;

    if(isWarn) warn(" - - - - Connecting to: \"" + device + "\"...\n");
    else info(" - - - - Connecting to: \"" + device + "\"...\n");

    InferenceEngine::ExecutableNetwork result;

    try {
        result = core.LoadNetwork(net, device);
        *ok = true;
        success(" - - - - Connected to \"" + device + "\".\n");
    } catch (...) {
        warn(" - - - - Failed to connect to device: \"" + device + "\".\n");
    }

    return result;
}

InferenceEngine::ExecutableNetwork NetworkThread::createExecutable(const InferenceEngine::CNNNetwork & net, const string & preferredDeviceType, bool *ok) const {
    InferenceEngine::ExecutableNetwork result;

    info(" - - - - Looking for available devices...\n");
    vector<string> availableDevices = core.GetAvailableDevices();

    info(" - - - - Looking for preferred device type \"" + preferredDeviceType + "\"...\n");
    string chosenDevice = getSimilarDevice(preferredDeviceType, availableDevices);

    if(chosenDevice != ""){
        success(" - - - - Preferred device type found: \"" + chosenDevice + "\".\n");
        result = tryToLoadNetwork(net, chosenDevice, false, ok);
        if(!*ok) warn(" - - - - Trying other available devices...\n");
    }
    else{
        error(" - - - - Couldn't find preferred device type.\n");
        warn(" - - - - Trying other available devices...\n");
    }

    if(!*ok){
        for(string dev : availableDevices){
            result = tryToLoadNetwork(net, dev, true, ok);
            if(ok)
                break;
        }
    }

    if(*ok){
        *ok = true;
        return result;
    }
    error(" - - Couldn't connect to any found device.\n");
    return result;
}

NetworkThread::CNN NetworkThread::loadNetwork(const string & modelPath, const string & preferredDeviceType, bool *ok, bool preprocessF,  InputInfo preprocess) const {
    info(" - - Loading \"" + modelPath + "\" to Inference Engine...\n");

    CNN net;
    *ok = false;

    if(modelPath == ""){
        error(" - - - Model's path can't be empty.\n");
        return net;
    }

    vector<string> availableDevices = core.GetAvailableDevices();
    if(!availableDevices.size()){
        error(" - - - No available devices found for this model.\n");
        return net;
    }

    try {
        info(" - - - Reading model file...\n");
        net.network = core.ReadNetwork(modelPath);
    } catch (...) {
        error(" - - - Couldn't read given model path.\n");
        return net;
    }

    info(" - - - Getting i/o info...\n");
    net.inputInfo = net.network.getInputsInfo();
    if(preprocessF){
        for(auto i : net.inputInfo){
            i.second->getPreProcess().setColorFormat(preprocess.format);
            i.second->setPrecision(preprocess.prec);
            i.second->setLayout(preprocess.lay);
        }
    }
    net.outputInfo = net.network.getOutputsInfo();

    info(" - - - Creating executable network...\n");
    net.executableNetwork = createExecutable(net.network, preferredDeviceType, ok);

    if(*ok)
        success(" - - Network loaded to Inference Engine.\n");
    return net;
}
