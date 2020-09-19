#include "eventdetectorthread.h"

#include <QCoreApplication>

#include "messages.h"

EventDetectorThread::~EventDetectorThread() {
    info("Stopping event detector...\n");

    info(" - Stopping audio input...\n");
    if(audioInput != nullptr){
        audioInput->stop();
        audioInput->reset();
        delete audioInput;
    }
    info(" - Closing audio buffer...\n");
    audioBuf.close();

    success("Event detector stopped.\n");
}

string EventDetectorThread::networkResponseToString(const vector<long double> & vals){
    string result;
    int index = distance(vals.begin(), max_element(vals.begin(), vals.end()));
    switch(index){
    case 0:
        result = "Idle";
        break;
    case 1:
        result = "Knock";
        break;
    case 2:
        result = "Scrub";
        break;
    default:
        result = "Unknown";
        break;
    }
    return result;
}

QAudioDeviceInfo EventDetectorThread::getAudioDeviceInfo(const QString & name) {
    QAudioDeviceInfo device;
    QList<QAudioDeviceInfo> devices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
    for(int i = 0; i < devices.size(); ++i) {
        if(devices.at(i).deviceName() == name) {
            return devices.at(i);
        }
    }

    devices = QAudioDeviceInfo::availableDevices(QAudio::AudioOutput);
    for(int i = 0; i < devices.size(); ++i) {
        if(devices.at(i).deviceName() == name) {
            return devices.at(i);
        }
    }

    return device;
}

void EventDetectorThread::slideWindowByPeriod(){
    for(unsigned int i = bytesPerPeriod; i < slidingWindow.size(); i++){
        slidingWindow[i - bytesPerPeriod] = slidingWindow[i];
    }
}

void EventDetectorThread::insertPeriodToWindow(){
    for(unsigned int i = slidingWindow.size() - bytesPerPeriod, j = 0; i < slidingWindow.size(); i++, j++){
        slidingWindow[i] = audioBuf.buffer()[j];
    }
}

void EventDetectorThread::clearWindow(MatrixMath::vec2d & v){
    // for each row
    for(unsigned int i = 0; i < v.size(); i++){
        // find it's lowest value
        long double lowest = numeric_limits<long double>::max();
        for(unsigned int j = 0; j < v[i].size(); j++){
            if(v[i][j] < lowest)
                lowest = v[i][j];
        }
        // replace first n columns with this lowest value
        long double denominator = conf.windowDuration / (long double)v[i].size();
        if(denominator == 0)
            continue;
        unsigned int columnsToClear = round(conf.clearedWindowPartDuration / denominator);
        for(unsigned int j = 0; j < columnsToClear; j++){
            v[i][j] = lowest;
        }

    }
}

vector<long double> EventDetectorThread::networkRequest(const vector<vector<long double>> & data){
    vector<long double> result;

    // prepare network request
    InferenceEngine::InferRequest request = eventDetectorNetwork.executableNetwork.CreateInferRequest();

    // prepare input
    string inputName = eventDetectorNetwork.inputInfo.begin()->first;
    InferenceEngine::MemoryBlob::Ptr input = dynamic_pointer_cast<InferenceEngine::MemoryBlob>(request.GetBlob(inputName));
    {
        const InferenceEngine::LockedMemory<void> memLocker = input->wmap();
        float *inputBuf = memLocker.as<float*>();
        for(unsigned int i = 0, k = 0; i < data.size(); i++){
            for(unsigned int j = 0; j < data[i].size(); j++){
                inputBuf[k] = data[i][j];
                k++;
            }
        }
        request.SetBlob(inputName, input);
    }

    request.Infer();

    // read output
    string outputName = eventDetectorNetwork.outputInfo.begin()->first;
    InferenceEngine::MemoryBlob::Ptr output = dynamic_pointer_cast<InferenceEngine::MemoryBlob>(request.GetBlob(outputName));
    {
        const InferenceEngine::LockedMemory<const void> memLocker = output->rmap();
        const float *outputBuf = memLocker.as<const float*>();
        for(unsigned int i = 0; i < output->size(); i++){
            result.push_back(outputBuf[i]);
        }
    }
    return result;
}

bool EventDetectorThread::runStep(){
    // record audio
    audioBuf.buffer().clear();
    audioBuf.reset();

    while(audioBuf.buffer().size() < bytesPerPeriod){
        QCoreApplication::processEvents();
    }

    // transform buffer:
    // slide it so there will be space for recorded audio
    // insert recorded audio to buffer
    // clear n first samples to "cheat" network so it will
    // predict events on smaller window than sizeOfSlidingWindow
    slideWindowByPeriod();
    insertPeriodToWindow();

    // get spectogram of sliding window
    MatrixMath::vec2d spectogram;
    spectogram = audioProcessor.processBuffer(slidingWindow);

    clearWindow(spectogram);

    // make decision
    vector<long double> res = networkRequest(spectogram);
    string state = networkResponseToString(res);

    // process decision
    // eliminate noise from the network
    // by looking at the previous result and state
    // sometimes it happens that knock is detected between scrubs
    // so we need to ignore any knock that happens just after successfully detected scrub
    // we also need to detect knock only if transition knock -> idle is detected
    // as it may happen that begining of scrub will be detected as knock
    string result = "Idle";
    if(state == "Scrub" || (state == "Knock" && previousResult == "Scrub"))
        result = "Scrub";
    else if(state == "Idle" && previousState == "Knock")
        result = "Knock";

    if(result != previousResult){
        emit touchStateChanged(QString::fromStdString(result));
    }

    previousState = state;
    previousResult = result;

    emit visualsChanged(spectogram);

    return true;
}

bool EventDetectorThread::start(const Config & threadConf, const QAudioFormat & audioFormat, const AudioProcessor::config & processorConfig){
    info("Starting event detector thread...\n");
    if(isRunning()){
        warn("Tried to run event detector thread but it's already running. Ignored.\n");
        return false;
    }

    bool ok;
    info(" - Loading event detection network...\n");
    eventDetectorNetwork = loadNetwork(threadConf.modelName, threadConf.preferredDeviceType, &ok);
    if(!ok)
        return false;

    conf = threadConf;

    info(" - Creating audio input device and opening audio buffer...\n");
    audioInput = new QAudioInput(getAudioDeviceInfo(QString::fromStdString(conf.device)), audioFormat, this);
    audioBuf.open(QIODevice::ReadWrite);
    audioProcessor.setConfig(processorConfig);

    info(" - Parsing config params...\n");
    slidingWindow.clear();
    sizeOfSlidingWindow = processorConfig.sampleRate * processorConfig.bytesPerSample * processorConfig.numberOfChannels * conf.windowDuration*0.001;
    bytesPerPeriod = processorConfig.sampleRate * processorConfig.bytesPerSample * processorConfig.numberOfChannels * conf.samplingPeriod*0.001;
    slidingWindow.resize(sizeOfSlidingWindow);

    info(" - Starging audio input...\n");
    audioInput->start(&audioBuf);
    QThread::start();

    success("Event detector thread has started.\n");
    return true;
}
