#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>
#include <thread>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    /**
     * @brief Class constructor.
     * @param parent Parent of this object.
     */
    explicit MainWindow(QWidget *parent = nullptr);

    /**
     * @brief Class destructor.
     */
    ~MainWindow();

private:
    Ui::MainWindow *ui; //!< UI objects from mainwindow.ui

    std::atomic<bool> killFlag; //!< On program end kill camera thread.
    std::mutex UiMtx; //!< Blocks access to ui objects.
    std::thread cameraThread; //!< Do stuff with camera, perform skew op etc.

    cv::Mat sourceImg; //!< Original frame from camera with skew points.
    cv::Mat targetImg; //!< Frame after skew operation with skew points in the corners.

    /**
     * @brief Convert OpenCV Mat into QPixmap.
     * @param src Source image.
     * @return QPixmap.
     */
    QPixmap MatToPixmap(cv::Mat src);

    /**
     * @brief Initialize default values of spinboxes based on first received frame from camera.
     */
    void initSpinBoxValues();

    /**
     * @brief Update spinbox properties to make sure that top left corner won't go beyong top right corner etc.
     */
    void updateSpinBoxProperties();

    /**
     * @brief Get frame from attached camera and perform skew transformation on it.
     */
    void getNewFrame();

    /**
     * @brief Draw skew points on source frame on positions described by eight spin boxes
     *  and draw skew points on each corner of transformed frame.
     */
    void drawSkewPoints();

    /**
     * @brief Transform cv::Mat into QPixmap and display it on UI.
     */
    void displayFrames();

    /**
     * @brief Do all the stuff with camera image processing in separate thread.
     */
    void cameraThreadFcn();

private slots:
    /**
     * @brief Update spinbox properties to make sure that top left corner won't go beyong top right corner etc.
     * @param unused Unused.
     */
    void spinBox_changed(int unused = 0);

    /**
     * @brief Display prompt to get filename and if it's valid then save skew info to filename.
     */
    void on_pushButton_clicked();
};

#endif // MAINWINDOW_H
