#ifndef POINTERTRACKERTHREAD_H
#define POINTERTRACKERTHREAD_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "networkthread.h"
#include "messages.h"

class PointerTrackerThread : public NetworkThread
{
    Q_OBJECT
public:
    /**
     * @brief Config.
     */
    struct Config{
        int cameraId;
        bool useIpCam;
        string IPCamURL;
        int screenWidth;
        int screenHeight;
        string embeddingSourceModelName;
        string embeddingSourcePreferredDeviceType;
        string embeddingExemplarModelName;
        string embeddingExemplarPreferredDeviceType;
        string crossoverPreferredDeviceType;
        int xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr;
    };

private:
    typedef InferenceEngine::SizeVector NetDims;
    typedef InferenceEngine::Blob::Ptr EmbeddingBlob;

    const NetworkThread::InputInfo preprocess = {
        InferenceEngine::Precision::U8,
        InferenceEngine::Layout::NHWC,
        InferenceEngine::ColorFormat::RGB
    }; //!< Preprocess embedding input to make it compatible with cv::Mat object.

    bool fstScan; //!< Required for initialization.
    Config conf; //!< Config for this thread.

    // camera stuff
    cv::Mat cameraFrame; //!< Camera's recent captured frame is stored here. Access using accessCamera().
    mutex cameraFrameMutex; //!< Used to access cameraFrame.
    promise<void> cameraExitSignal; //!< Set value to kill camera thread.

    // nnet stuff
    CNN embeddingSourceSubnet; //!< Embedding network to use with source frame.
    vector<InferenceEngine::Blob::Ptr> exemplarFilters; //!< Results of embeddings with exemplars.
    CNN crossCorelationLayer; //!< Cross corelation layer

    /**
     * @brief Access recent frame from camera.
     * @return Recent frame from camera.
     */
    cv::Mat accessCamera();

    /**
     * @brief Warp perspective of given frame based on ints from config.
     * @param frame Frame to warp.
     * @return Warped frame.
     */
    cv::Mat warpPerspective(const cv::Mat & frame) const;

    /**
     * @brief Apply predictions to frame in form of inferno heatmap.
     * @param frame Frame to apply heatmap to.
     * @param predictions Predictions to apply.
     * @return Frame with heatmap applied.
     */
    cv::Mat applyHeatmap(const cv::Mat & frame, const cv::Mat & predictions) const;

    /**
     * @brief Get x and y values from area of prediction map with biggest confidence.
     * @param predictions Prediction map to search for x and y.
     * @param x Found x is stored here.
     * @param y Founy y is stored here.
     */
    void getCursorCoords(const cv::Mat & predictions, int & x, int & y) const;

    /**
     * @brief Get blob of data with result of embedding given image with given network.
     * @param net Embedding network.
     * @param img Image to perform embedding on.
     * @return Blob of data
     */
    EmbeddingBlob getEmbeddingBlob(CNN & net, const cv::Mat & img) const;

    /**
     * @brief Get input dimensions for given embedding network.
     * @param net Network to get input dimensions from.
     * @return Input dimensions of given network.
     */
    NetDims getEmbeddingInputDims(const CNN & net) const;

    /**
     * @brief Get output dimensions for given embedding network.
     * @param net Network to get output dimensions from.
     * @return Output dimensions of given network.
     */
    NetDims getEmbeddingOutputDims(const CNN & net) const;

    /**
     * @brief Generate cross corelation layer based on dimensions of source frame and exemplar.
     * @param embeddingDims Dimensions of output of embedding with exemplar.
     * @param ok Set to false on fail.
     * @return Generated cross corelation layer.
     */
    CNN generateCrossCorelationLayer(const vector<size_t> & embeddingDims, bool *ok = nullptr) const;

    /**
     * @brief Generate exemplars and cross corelation layer.
     * @return True on success.
     */
    bool finalizeSiameseNetwork();

    /**
     * @brief Perform network request.
     * @param input Input to perform request on.
     * @param filter Filter.
     * @return Score map.
     */
    cv::Mat networkRequest(const cv::Mat & input, const EmbeddingBlob & filter);

    /**
     * @brief Wait given number of milliseconds but update GUI with frames from camera.
     * @param circlePosition Where to draw circle. Set either coord to -1 to ignore.
     * @param t Time to wait in ms.
     * @return Last frame received from camera.
     */
    cv::Mat activeWait(const cv::Point & circlePosition, int t);

    /**
     * @brief One step of pointer tracker thread.
     * @return True if successful, false to kill loop.
     */
    bool runStep() override;

public:
    ~PointerTrackerThread() override;

    /**
     * @brief Start pointer tracking thread.
     * @param Configuration of the thread.
     * @return True if thread started.
     */
    bool start(Config threadConf);

signals:
    /**
     * @brief Signal emited to GUI to inform it about new cursor position.
     * @param x X position of cursor.
     * @param y Y position of cursor.
     */
    void cursorPositionChanged(int x, int y);

    /**
     * @brief Signal emited to GUI to update live preview.
     * @param frame Frame to change preview to.
     */
    void visualsChanged(const cv::Mat & frame);
};

#endif // POINTERTRACKERTHREAD_H
