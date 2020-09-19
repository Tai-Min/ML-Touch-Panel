#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <opencv2/core/core.hpp>

#include "audioprocessor.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    /**
     * @brief Convert simple matrix to spectogram in for of heatmap.
     * @param v Spectogram to convert.
     * @param width Width of resulting pixmap.
     * @param height Height of resulting pixmap.
     * @return Spectogram in form of pixmap.
     */
    static QPixmap spectogramToPixmap(const MatrixMath::vec2d & v, unsigned int width, unsigned int height);

    /**
     * @brief Convert cv::Mat to pixmap.
     * @param mat Mat to convert.
     * @param width Width of resulting pixmap.
     * @param height Height of resulting pixmap.
     * @return Mat in form of pixmap.
     */
    static QPixmap CvMatToPixmap(const cv::Mat & mat, unsigned int width, unsigned int height);

public slots:
    /**
     * @brief Perform mouse click / draw on given position. Also update GUI label
     * @param state
     */
    void changeCursorState(const QString & state);

    /**
     * @brief Update spectogram's label with given vector.
     * @param spectogram Spectogram to update label with.
     */
    void updateSpectogram(const MatrixMath::vec2d & spectogram);

    /**
     * @brief Save received cursor position.
     * @param x X position.
     * @param y Y position.
     */
    void changeCursorPosition(int x, int y);

    /**
     * @brief Update preview label.
     * @param frame Frame to update preview label with.
     */
    void updatePreview(const cv::Mat & frame);
};

#endif // MAINWINDOW_H
