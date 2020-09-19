#include <QApplication>
#include <QCommandLineParser>

#include "eventdetectorthread.h"
#include "pointertrackerthread.h"
#include "messages.h"
#include "mainwindow.h"

#include "thirdparty/json/single_include/nlohmann/json.hpp"
using json = nlohmann::json;

#include <fstream>
using namespace std;

Q_DECLARE_METATYPE(MatrixMath::vec2d);
Q_DECLARE_METATYPE(cv::Mat);

/* main */
int main(int argc, char *argv[]);

/* helpers */
/**
 * @brief Run event detector thread.
 * @param conf Config in JSON format.
 * @param eventDetector Object to run.
 * @return 0 on success, -1 on fail.
 */
int runEventDetector(const json & conf, EventDetectorThread & eventDetector);

/**
 * @brief Run pointer tracker thread.
 * @param conf Config in JSON format.
 * @param pointerTracker Object to run.
 * @return 0 on success, -1 on fail.
 */
int runPointerTracker(const json & conf, PointerTrackerThread & pointerTracker);

/**
 * @brief Run application.
 * @param conf Config in JSON format.
 * @param app Application object.
 * @return Return code.
 */
int run(const json & conf, QCoreApplication & app);

/**
 * @brief Check config for all required symbols.
 * @param conf Config to check.
 * @return True if config is valid.
 */
bool isValidConfig(const json & conf);

/**
 * @brief Load config from given path.
 * @param confPath Path to config.
 * @param ok Set to false if something gone wrong.
 * @return Config object or undefined if ok is false.
 */
json loadConfig(const string & confPath = "config.json", bool *ok = nullptr);

/* definitions main */
int main(int argc, char *argv[])
{
    qRegisterMetaType<MatrixMath::vec2d>();
    qRegisterMetaType<cv::Mat>();

    QApplication app(argc, argv);
    QApplication::setApplicationName("ML touch panel runtime");
    QApplication::setApplicationVersion("1.0");

    // parse arguments
    QCommandLineParser argsParser;
    argsParser.setApplicationDescription("Runtime for ML touch panel.");
    argsParser.addHelpOption();
    argsParser.addVersionOption();
    argsParser.addOptions({
                              {{"c", "config"}, "Configuration file. By default it's config.json in application's directory.", "config.json"}
                          });
    argsParser.process(app);

    info("Application starting...\n");

    // load and validate json config
    string confPath = (argsParser.isSet("config") ? argsParser.value("config").toStdString() : "config.json");
    info("Loading configuration file " + confPath + "...\n");

    bool ok;
    json conf = loadConfig(confPath, &ok);
    if(!ok)
        return -1;

    if(!isValidConfig(conf))
        return -1;

    // run application
    return run(conf, app);
}

int runEventDetector(const json & conf, EventDetectorThread & eventDetector){
    json detectorConfJsn = conf["eventDetector"];
    EventDetectorThread::Config detectorConf = {
        detectorConfJsn["audioInput"]["deviceName"],
        detectorConfJsn["modelPath"],
        detectorConfJsn["preferredDeviceType"],
        detectorConfJsn["windowDuration"].get<unsigned int>(),
        detectorConfJsn["clearedWindowPartDuration"].get<unsigned int>(),
        detectorConfJsn["samplingPeriod"].get<unsigned int>()
    };

    json formatConfJsn = detectorConfJsn["audioInput"];
    QAudioFormat audioFormat;
    audioFormat.setSampleRate(formatConfJsn["sampleRate"].get<unsigned int>());
    audioFormat.setChannelCount(formatConfJsn["channelCount"].get<unsigned int>());
    audioFormat.setSampleSize(formatConfJsn["sampleSize"].get<unsigned int>());
    audioFormat.setCodec("audio/pcm");
    audioFormat.setByteOrder(QAudioFormat::LittleEndian);
    audioFormat.setSampleType(QAudioFormat::UnSignedInt);

    json processorConfJsn = conf["eventDetector"]["audioProcessor"];
    AudioProcessor::config processorConfig = {
        formatConfJsn["sampleSize"].get<unsigned int>()/8,
        formatConfJsn["channelCount"].get<unsigned int>(),
        formatConfJsn["sampleRate"].get<unsigned int>(),
        processorConfJsn["preEmphasisCoeff"].get<long double>(),
        processorConfJsn["frameSize"].get<unsigned int>(),
        processorConfJsn["frameStride"].get<unsigned int>(),
        processorConfJsn["DFTs"].get<unsigned int>(),
        processorConfJsn["filterBanks"].get<unsigned int>(),
        processorConfJsn["useMFCC"].get<bool>(),
        processorConfJsn["MFCC"]["firstCoeffToKeep"].get<unsigned int>(),
        processorConfJsn["MFCC"]["lastCoeffToKeep"].get<unsigned int>(),
        processorConfJsn["MFCC"]["useSinLift"].get<bool>(),
        processorConfJsn["MFCC"]["sinLift"]["cepstralLifters"].get<unsigned int>(),
        processorConfJsn["normalizeData"].get<bool>(),
        processorConfJsn["rescaleData"].get<bool>(),
        processorConfJsn["rescale"]["min"].get<long double>(),
        processorConfJsn["rescale"]["max"].get<long double>(),
    };

    if(!eventDetector.start(detectorConf, audioFormat, processorConfig))
        return -1;

    return 0;
}

int runPointerTracker(const json & conf, PointerTrackerThread & pointerTracker){
    json trackerConfJsn = conf["pointerTracker"];
    PointerTrackerThread::Config trackerConf = {
        trackerConfJsn["cameraId"].get<int>(),
        trackerConfJsn["useIPCam"].get<bool>(),
        trackerConfJsn["IPCamURL"],
        trackerConfJsn["screenWidth"].get<int>(),
        trackerConfJsn["screenHeight"].get<int>(),
        trackerConfJsn["embeddingSourceModelPath"],
        trackerConfJsn["embeddingSourcePreferredDeviceType"],
        trackerConfJsn["embeddingExemplarModelPath"],
        trackerConfJsn["embeddingExemplarPreferredDeviceType"],
        trackerConfJsn["crossoverPreferredDeviceType"],
        trackerConfJsn["transformation"]["xtl"].get<int>(),
        trackerConfJsn["transformation"]["ytl"].get<int>(),
        trackerConfJsn["transformation"]["xtr"].get<int>(),
        trackerConfJsn["transformation"]["ytr"].get<int>(),
        trackerConfJsn["transformation"]["xbl"].get<int>(),
        trackerConfJsn["transformation"]["ybl"].get<int>(),
        trackerConfJsn["transformation"]["xbr"].get<int>(),
        trackerConfJsn["transformation"]["ybr"].get<int>()
    };

    if(!pointerTracker.start(trackerConf))
        return -1;

    return 0;
}

int run(const json & conf, QCoreApplication & app){
    MainWindow w;
    bool cleanExit = false;
    EventDetectorThread eventDetector;
    PointerTrackerThread pointerTracker;

    QObject::connect(&eventDetector, &EventDetectorThread::visualsChanged, &w, &MainWindow::updateSpectogram);
    QObject::connect(&eventDetector, &EventDetectorThread::touchStateChanged, &w, &MainWindow::changeCursorState);

    QObject::connect(&pointerTracker, &PointerTrackerThread::visualsChanged, &w, &MainWindow::updatePreview);
    QObject::connect(&pointerTracker, &PointerTrackerThread::cursorPositionChanged, &w, &MainWindow::changeCursorPosition);


    // in case event detector crashes
    QObject::connect(&eventDetector, &QThread::finished, [&](){
        if(!cleanExit){
            error("Event detector stopped unexpectedly.\n");
            app.exit(-1);
        }
    });

    // in case pointer tracker crashes
    QObject::connect(&pointerTracker, &QThread::finished, [&](){
        if(!cleanExit){
            error("Pointer tracker stopped unexpectedly.\n");
            app.exit(-1);
        }
    });

    // clean quit
    QObject::connect(&app, &QApplication::aboutToQuit, [&](){
        info("Application exiting...\n");
        cleanExit = true;
    });

    if (runEventDetector(conf, eventDetector) != 0) return -1;
    if (runPointerTracker(conf, pointerTracker) != 0) return -1;

    w.show();
    return app.exec();
}

bool isValidConfig(const json & conf){
    return true;
}

json loadConfig(const string & confPath, bool *ok){
    *ok = false;

    json conf;
    ifstream confFile(confPath);
    if(!confFile.is_open()){
        error("Couldn't open configuration file.\n");
        return conf;
    }
    try {
        confFile >> conf;
    } catch (...) {
        error("Failed to parse config file.\n");
        return conf;
    }

    confFile.close();

    *ok = true;
    return conf;
}
