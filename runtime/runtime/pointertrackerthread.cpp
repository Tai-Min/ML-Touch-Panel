#include "pointertrackerthread.h"
#include "ngraph/opsets/opset3.hpp"

PointerTrackerThread::~PointerTrackerThread() {
    info("Stopping pointer tracker...\n");
    if(!isRunning()){
        return;
    }

    // stop camera thread
    info(" - Stopping camera thread...\n");
    cameraExitSignal.set_value();
    success(" - Camera thread stopped.\n");

    success("Pointer tracker stopped.\n");
}

cv::Mat PointerTrackerThread::accessCamera(){
    unique_lock<mutex> lk(cameraFrameMutex);
    return cameraFrame;
}

cv::Mat PointerTrackerThread::warpPerspective(const cv::Mat & frame) const {
    cv::Point2f source[4] = {cv::Point(conf.xtl,conf.ytl),
                             cv::Point(conf.xtr,conf.ytr),
                             cv::Point(conf.xbl,conf.ybl),
                             cv::Point(conf.xbr,conf.ybr)};

    cv::Point2f target[4] = {cv::Point(0,0),
                             cv::Point(frame.cols,0),
                             cv::Point(0,frame.rows),
                             cv::Point(frame.cols,frame.rows)};

    cv::Mat transformation = cv::getPerspectiveTransform(source,target);
    cv::Mat result;
    cv::warpPerspective(frame, result, transformation, cv::Size(frame.cols, frame.rows));
    return result;
}

cv::Mat PointerTrackerThread::applyHeatmap(const cv::Mat & frame, const cv::Mat & predictions) const {
    cv::Mat predsU8;
    cv::normalize(predictions, predsU8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(predsU8, predsU8, cv::COLORMAP_INFERNO);
    cv::Mat result;
    cv::addWeighted(frame, 0.6, predsU8, 0.4, 0, result);
    return result;
}

void PointerTrackerThread::getCursorCoords(const cv::Mat & predictions, int & x, int & y) const {
    double min, max;
    int minIdx[2], maxIdx[2];
    cv::minMaxIdx(predictions, &min, &max, minIdx, maxIdx);

    x = maxIdx[1];
    y = maxIdx[0];
}

PointerTrackerThread::EmbeddingBlob PointerTrackerThread::getEmbeddingBlob(CNN & exemplarNet, const cv::Mat &exemplar) const {
    // prepare inputs
    InferenceEngine::InferRequest embeddingRequest = exemplarNet.executableNetwork.CreateInferRequest();
    InferenceEngine::TensorDesc inDesc (
                InferenceEngine::Precision::U8,
                getEmbeddingInputDims(exemplarNet),
                InferenceEngine::Layout::NHWC);
    InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<uint8_t>(inDesc, exemplar.data, exemplar.total() * exemplar.elemSize());
    embeddingRequest.SetBlob(exemplarNet.inputInfo.begin()->first, inBlob);

    // do inference
    embeddingRequest.Infer();

    // return output blob
    return embeddingRequest.GetBlob(exemplarNet.outputInfo.begin()->first);
}

PointerTrackerThread::NetDims PointerTrackerThread::getEmbeddingInputDims(const CNN & net) const {
    return net.network.getInputsInfo().begin()->second->getTensorDesc().getDims();
}

PointerTrackerThread::NetDims PointerTrackerThread::getEmbeddingOutputDims(const CNN & net) const {
    return net.network.getOutputsInfo().begin()->second->getTensorDesc().getDims();
}

NetworkThread::CNN PointerTrackerThread::generateCrossCorelationLayer(const vector<size_t> & embeddingDims, bool *ok) const {
    info(" - - Loading corelation layer to Inference Engine...\n");

    // create inputs for cross corelation
    info(" - - - Generating inputs to cross corelation...\n");
    auto input = make_shared<ngraph::opset3::Parameter>(
                ngraph::element::Type_t::f32, ngraph::Shape(getEmbeddingOutputDims(embeddingSourceSubnet)));
    input->set_friendly_name("input");

    auto filter = make_shared<ngraph::opset3::Parameter>(
                ngraph::element::Type_t::f32, ngraph::Shape(embeddingDims));
    filter->set_friendly_name("filter");

    // conv layer
    info(" - - - Generating cross corelation...\n");
    std::shared_ptr<ngraph::Node> conv = make_shared<ngraph::opset3::Convolution>(
                input->output(0),
                filter->output(0),
                ngraph::Strides({1,1}),
                ngraph::CoordinateDiff({0,0}),
                ngraph::CoordinateDiff({0,0}),
                ngraph::Strides({1,1}));
    ngraph::NodeVector ops = { input, filter, conv};

    // validate
    info(" - - - Validating cross corelation...\n");
    ngraph::validate_nodes_and_infer_types(ops);

    // create ngraph Function object from inputs and conv
    info(" - - - Creating ngraph function...\n");
    shared_ptr<ngraph::Function> ng_function = make_shared<ngraph::Function>(ngraph::OutputVector({conv}), ngraph::ParameterVector{ input, filter });

    // create network from ngraph Function object
    info(" - - - Creating network object...\n");
    InferenceEngine::CNNNetwork net(ng_function);

    CNN result;

    // IO info
    info(" - - - Getting i/o info...\n");
    result.network = net;
    result.inputInfo = net.getInputsInfo();
    result.outputInfo = net.getOutputsInfo();

    // create executable
    info(" - - - Creating executable network...\n");
    result.executableNetwork = createExecutable(net, conf.crossoverPreferredDeviceType, ok);

    if(*ok){
        success(" - - Cross corelation loaded to Inference Engine.\n");
        success(" - Cross corelation layer generated.\n");
    }
    else{
        error(" - Failed to generate cross corelation layer.\n");
    }

    return result;
}

bool PointerTrackerThread::finalizeSiameseNetwork(){
    info("Initializing exemplars...\n");

    // load exemplar subnet
    info(" - Initializing exemplar embedding subnet...\n");
    bool ok;
    CNN embeddingExemplarSubnet = loadNetwork(conf.embeddingExemplarModelName, conf.embeddingExemplarPreferredDeviceType, &ok, true, preprocess);
    if(!ok)
        return false;

    // generate crossover layer
    info(" - Generating cross corelation layer...\n");
    crossCorelationLayer = generateCrossCorelationLayer(getEmbeddingOutputDims(embeddingExemplarSubnet), &ok);
    if(!ok)
        return false;

    // get frame for it's dimensions
    cv::Mat frame = activeWait({0,0}, 10);
    int frameWidth = frame.size().width;
    int frameHeight = frame.size().height;

    // wait a bit for user to follow instructions
    info("Generating exemplars...\n");
    info("Place your cursor on white dot.\n");
    frame = activeWait({frameWidth/2, frameHeight/2}, 5000);

    NetDims exemplarDims = getEmbeddingInputDims(embeddingExemplarSubnet);
    int exemplarWidth = exemplarDims[3];
    int exemplarHeight = exemplarDims[2];

    // generate some exemplar filters
    //for(int i = -2; i <= 2; i+=2){
        // like in original paper
        int width = exemplarWidth * pow(1.1, 1);
        int height = exemplarHeight * pow(1.1, 1);

        cv::Mat exemplar;
        // copy center of frame to exemplar
        frame({int(frameWidth/2 - width/2), int(frameHeight/2 - height/2),width, height}).copyTo(exemplar);

        // resize it to match network's input
        cv::resize(exemplar, exemplar, {(int)exemplarDims[3], (int)exemplarDims[2]});

        // add generated filter blob to vector
        exemplarFilters.push_back(getEmbeddingBlob(embeddingExemplarSubnet, exemplar));

        // show it to user for some time
        emit visualsChanged(exemplar);
        QThread::sleep(1);
    //}
    success("Exemplars generated.\n");
    return true;
}

cv::Mat PointerTrackerThread::networkRequest(const cv::Mat & sourceFrame, const EmbeddingBlob & filter){
    // cross corelation
    InferenceEngine::InferRequest xcorRequest = crossCorelationLayer.executableNetwork.CreateInferRequest();

    // prepare inputs for cross corelation
    xcorRequest.SetBlob("input", getEmbeddingBlob(embeddingSourceSubnet, sourceFrame));
    xcorRequest.SetBlob("filter", filter);

    // create empty score map
    NetDims dims = crossCorelationLayer.outputInfo.begin()->second->getTensorDesc().getDims();
    cv::Mat res = cv::Mat::zeros(dims[2], dims[3], CV_32FC1);

    // prepare output
    InferenceEngine::TensorDesc resDesc (
                InferenceEngine::Precision::FP32,
                crossCorelationLayer.outputInfo.begin()->second->getTensorDesc().getDims(),
                InferenceEngine::Layout::NCHW);
    InferenceEngine::Blob::Ptr resBlob = InferenceEngine::make_shared_blob<float>(resDesc, (float*)res.data, dims[2] * dims[3]);

    string outputName = crossCorelationLayer.outputInfo.begin()->first;
    xcorRequest.SetOutput(InferenceEngine::BlobMap({{outputName, resBlob}}));

    // do inference with cross corelation
    xcorRequest.Infer();

    return res;
}

cv::Mat PointerTrackerThread::activeWait(const cv::Point & circlePosition, int t){
    cv::Mat frame;

    auto start = chrono::high_resolution_clock::now();
    auto now = chrono::high_resolution_clock::now();
    auto time = now - start;

    // do it for given time in ms
    while(time / chrono::milliseconds(1) < t){
        frame = cv::Mat();

        // get valid frame
        while(frame.empty()){
            frame = accessCamera();
            QThread::msleep(10);
        }

        // warp it to get only board
        frame = warpPerspective(frame);

        // display warped frame through gui thread
        cv::Mat displayFrame;
        frame.copyTo(displayFrame);
        if(circlePosition.x > -1 && circlePosition.y > -1)
            cv::circle(displayFrame, circlePosition, 7, {255,255,255}, -1);
        emit visualsChanged(displayFrame);

        time = chrono::high_resolution_clock::now() - start;
    }
    return frame;
}

bool PointerTrackerThread::runStep() {
    // do some initialization stuff
    if(fstScan){
        if(!finalizeSiameseNetwork())
            return false;
        fstScan = false;
    }

    // get new frame from camera
    cv::Mat frame = accessCamera();

    // ignore empties that may happen before camera initializes
    if(frame.empty())
        return true;

    // transform perspective so blackboard is centered and fills whole frame
    frame = warpPerspective(frame);

    NetDims inputDims = getEmbeddingInputDims(embeddingSourceSubnet);

    //resize frame to match network's input
    cv::resize(frame, frame, {(int)inputDims[3],(int)inputDims[2]}, cv::INTER_CUBIC);

    int cntr = 1;
    cv::Mat scoreMap;
    // get score map that is created from all partial score maps from all exemplar filters
    for(auto & filter : exemplarFilters){
        cv::Mat partialMap = networkRequest(frame, filter);
        if(cntr == 1){
            partialMap.copyTo(scoreMap);
        }
        else{
            cv::addWeighted(scoreMap, (cntr-1)/(double)cntr, partialMap, 1/(double)cntr, 0, scoreMap);
        }
        cntr++;
    }

    // resize score map to match frame's size
    cv::resize(scoreMap, scoreMap, frame.size(), cv::INTER_CUBIC);

    // get cursor coords from score map
    int x, y;
    getCursorCoords(scoreMap, x, y);
    int screenX = x*(double)(conf.screenWidth/(double)scoreMap.size().width)+1920;
    int screenY = y*(double)(conf.screenHeight/(double)scoreMap.size().height);
    emit cursorPositionChanged(screenX, screenY);

    // generate heatmap for gui
    cv::Mat heatmap = applyHeatmap(frame, scoreMap);
    cv::circle(heatmap, {x, y}, 3, {255,255,255}, -1);
    emit visualsChanged(heatmap);

    return true;
}

bool PointerTrackerThread::start(Config threadConf){
    info("Starting pointer tracker thread...\n");
    if(isRunning()){
        warn("Tried to run pointer tracker thread but it's already running. Ignored.\n");
        return false;
    }

    conf = threadConf;

    // load source subnet
    bool ok;
    info(" - Loading source embedding subnet...\n");
    embeddingSourceSubnet = loadNetwork(conf.embeddingSourceModelName, conf.embeddingSourcePreferredDeviceType, &ok, true, preprocess);
    if(!ok)
        return false;

    // create camera thread
    info(" - Creating and starting camera thread...\n");
    cameraExitSignal = promise<void>();
    class::thread cameraThread([=](){
        future<void> cameraExitFuture = cameraExitSignal.get_future();
        cv::VideoCapture cap;
        if(conf.useIpCam)
            cap.open(conf.IPCamURL);
        else
            cap.open(conf.cameraId);

        if(!cap.isOpened()){
            error("Couldn't open camera.\n");
            return;
        }

        while(cameraExitFuture.wait_for(chrono::microseconds(1)) == future_status::timeout){
            unique_lock<mutex> lk(cameraFrameMutex);
            cap.read(cameraFrame);
        }
        cap.release();
    });
    info(" - Detaching camera thread...\n");
    cameraThread.detach();

    fstScan = true;

    QThread::start();

    success("Pointer tracker thread has started.\n");
    return true;
}
