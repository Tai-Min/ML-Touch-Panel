#ifndef EVENTDETECTORTHREAD_H
#define EVENTDETECTORTHREAD_H

#include <QAudioInput>
#include <QBuffer>

#include "networkthread.h"
#include "audioprocessor.h"

class EventDetectorThread : public NetworkThread
{
    Q_OBJECT
public:
    /**
     * @brief Struct to store thread's config.
     */
    struct Config{
        string device;
        string modelName;
        string preferredDeviceType;
        unsigned int windowDuration;
        unsigned int clearedWindowPartDuration;
        unsigned int samplingPeriod;
    };

    /**
     * @brief Translate network's response to string.
     * @param vals Network's response.
     * @return Network response as string.
     */
    static string networkResponseToString(const vector<long double> & vals);
    static QAudioDeviceInfo getAudioDeviceInfo(const QString & name);

private:
    Config conf; //!< Config.

    unsigned int sizeOfSlidingWindow; //!< How big is sliding window in bytes.
    unsigned int bytesPerPeriod; //!< How big is one sampling period in bytes.
    AudioProcessor::byteVec slidingWindow; //!< Sliding window in bytes.

    string previousState = "Idle"; //!< Previous network's state. Changes every iteration.
    string previousResult = "Idle"; //!< Previous network's result. Changes depending on how state was processed.

    CNN eventDetectorNetwork; //!< Event detector network.
    AudioProcessor audioProcessor; //!< Audio processor to get spectogram.
    QBuffer audioBuf; //!< Buffer for audio samples.
    QAudioInput *audioInput = nullptr; //!< Input device.

    /**
     * @brief Slide window's vector to left by one sampling period.
     */
    void slideWindowByPeriod();

    /**
     * @brief Insert samples for recent sampling period to sliding window.
     */
    void insertPeriodToWindow();

    /**
     * @brief Clear n first columns from given matrix. Number of columns is defined by "clearedWindowPartDuration" JSON config.
     * @param v Matrix to clear columns in and result of operation.
     */
    void clearWindow(MatrixMath::vec2d & v);

    /**
     * @brief Perform event detection.
     * @param data Spectogram to process in network.
     * @return Result of event detection in form of three floating points [Idle, Knock, Scrub].
     */
    vector<long double> networkRequest(const vector<vector<long double>> & data);

    /**
     * @brief One event detector iteration.
     * @return True on success. False stops the thread.
     */
    bool runStep() override;

public:
    ~EventDetectorThread() override;

    /**
     * @brief Start event detector thread.
     * @param threadConf Thread config.
     * @param audioFormat Audio format.
     * @param processorConfig Config to postprocess audio data into spectogram.
     * @return True if thread started.
     */
    bool start(const Config & threadConf, const QAudioFormat & audioFormat, const AudioProcessor::config & processorConfig);

signals:
    /**
     * @brief Emited when event detector detects new state.
     * @param State to emit.
     */
    void touchStateChanged(const QString & state);

    /**
     * @brief Emit spectogram to show it in GUI.
     * @param Spectogram to emit.
     */
    void visualsChanged(const MatrixMath::vec2d & spectogram);
};

#endif // EVENTDETECTORTHREAD_H
