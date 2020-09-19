#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "messages.h"
#include <Windows.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setFixedSize(QSize(450,600));
}

MainWindow::~MainWindow(){
    delete ui;
}

QPixmap MainWindow::spectogramToPixmap(const MatrixMath::vec2d & v, unsigned int width, unsigned int height){
    const long double minValSrc = MatrixMath::minMatrix(v);
    const long double maxValSrc = MatrixMath::maxMatrix(v);
    const long double colorMax = 0; // red in HSV
    const long double colorMin = 240; // dark blue in HSV
    const long double a = (colorMax - colorMin)/(maxValSrc - minValSrc);
    const long double b = colorMin - a * minValSrc;
    QImage img = QImage(v[0].size(), v.size(), QImage::Format_RGB32);

    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
            QColor c = QColor::fromHsv(a*v[i][j] + b, 255, 255);
            img.setPixelColor(j, i, c);
        }
    }

    return QPixmap::fromImage(img.scaled(QSize(width, height)));
}

QPixmap MainWindow::CvMatToPixmap(const cv::Mat & src, unsigned int width, unsigned int height)
{
    QImage::Format format=QImage::Format_RGB888;
    QImage img(src.cols,src.rows,format);
    uchar *sptr,*dptr;
    int linesize=src.cols*3;

    for(int y=0;y<src.rows;y++){
        sptr=(uchar*)src.ptr(y);
        dptr=img.scanLine(y);
        memcpy(dptr,sptr,linesize);
    }

    return QPixmap::fromImage(img.rgbSwapped().scaled(width, height));
}

void MainWindow::changeCursorState(const QString & state){
    ui->touchStateLabel->setText(state);

    if(state== "Knock"){
        INPUT    Input={0};													// Create our input.
        Input.type        = INPUT_MOUSE;									// Let input know we are using the mouse.
        Input.mi.dwFlags  = MOUSEEVENTF_LEFTDOWN;							// We are setting left mouse button down.
        SendInput( 1, &Input, sizeof(INPUT) );								// Send the input.
        ZeroMemory(&Input,sizeof(INPUT));									// Fills a block of memory with zeros.
        Input.type        = INPUT_MOUSE;									// Let input know we are using the mouse.
        Input.mi.dwFlags  = MOUSEEVENTF_LEFTUP;								// We are setting left mouse button up.
        SendInput( 1, &Input, sizeof(INPUT) );
    }
    else if(state == "Scrub"){
        INPUT    Input={0};													// Create our input.
        Input.type        = INPUT_MOUSE;									// Let input know we are using the mouse.
        Input.mi.dwFlags  = MOUSEEVENTF_LEFTDOWN;							// We are setting left mouse button down.
        SendInput( 1, &Input, sizeof(INPUT) );
    }
    else{
        INPUT    Input={0};
        Input.type        = INPUT_MOUSE;									// Let input know we are using the mouse.
        Input.mi.dwFlags  = MOUSEEVENTF_LEFTUP;								// We are setting left mouse button up.
        SendInput( 1, &Input, sizeof(INPUT) );
    }
}

void MainWindow::updateSpectogram(const MatrixMath::vec2d & spectogram){
    QPixmap spectogramPixmap = spectogramToPixmap(spectogram, ui->spectogramLabel->width(), ui->spectogramLabel->height());
    ui->spectogramLabel->setPixmap(spectogramPixmap);
}

void MainWindow::changeCursorPosition(int x, int y){
    QCursor::setPos(x,y);
}

void MainWindow::updatePreview(const cv::Mat & mat){
    QPixmap heatmapPixmap = CvMatToPixmap(mat, ui->cameraView->width(), ui->cameraView->height());
    ui->cameraView->setPixmap(heatmapPixmap);
}
