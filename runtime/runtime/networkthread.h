#ifndef NETWORKTHREAD_H
#define NETWORKTHREAD_H

#include <QThread>
#include "inference_engine.hpp"

using namespace std;

class NetworkThread : public QThread
{
public:
    /**
     * @brief Structure that contains info about network, it's i/o and executable.
     */
    struct CNN{
        InferenceEngine::CNNNetwork network;
        InferenceEngine::InputsDataMap inputInfo;
        InferenceEngine::OutputsDataMap outputInfo;
        InferenceEngine::ExecutableNetwork executableNetwork;
    };

    /**
     * @brief Structure that contains info about network input's precision, layout and format.
     */
    struct InputInfo{
        InferenceEngine::Precision prec;
        InferenceEngine::Layout lay;
        InferenceEngine::ColorFormat format;
    };

    typedef vector<string> DeviceList;

private:
    /**
     * @brief Run infinite loop in new thread.
     */
    void run() override;

protected:
    static InferenceEngine::Core core; //!< Inference core.

    virtual ~NetworkThread() override;

    promise<void> threadExitSignal; //!< Promise that shall be fullified to kill this thread.

    /**
     * @brief Get similar device. Similar device for "CPU" could be "CPU" or "CPU1" etc.
     * @param dev Device to search for.
     * @param devices List of devices.
     * @return Similar device from given list or empty string if fail.
     */
    string getSimilarDevice(const string & dev, const DeviceList & devices) const;

    /**
     * @brief Try load given network to given device.
     * @param net Net model to load.
     * @param device Device to load network to.
     * @param isWarn If true then prints logs as warn instead of info.
     * @param ok Set to false if network was not loaded.
     * @return Executable network or undefined net if ok is set to false.
     */
    InferenceEngine::ExecutableNetwork tryToLoadNetwork(const InferenceEngine::CNNNetwork & net, const string & device, bool isWarn, bool *ok) const;

    /**
     * @brief Create executable network on preferred device or other device is preferred fails.
     * @param net Network to create executable from.
     * @param preferredDeviceType Peferred device type to load network to.
     * @param ok False on fail.
     * @return
     */
    InferenceEngine::ExecutableNetwork createExecutable(const InferenceEngine::CNNNetwork & net, const string & preferredDeviceType, bool *ok = nullptr) const;

    /**
     * @brief Load given network from xml file to preferred device or other device is preferred fails.
     * @param modelPath Path to model in xml format.
     * @param preferredDeviceType Device to load network to. If there is no similar device then network will be loaded to first available device.
     * @param ok Set to false if network couldn't be loaded.
     * @param preprocessF Whether use next param to preprocess inputs.
     * @param Struct that contains preprocessing info for inputs.
     * @return Network object or undefined network object if ok is set to false.
     */
    CNN loadNetwork(const string & modelPath, const string & preferredDeviceType, bool *ok = nullptr, bool preprocessF = false,  InputInfo preprocess = {}) const;

    /**
     * @brief Do one step in network thread.
     * @return True if success, fail stops the thread.
     */
    virtual bool runStep() = 0;
};

#endif // NETWORKTHREAD_H
