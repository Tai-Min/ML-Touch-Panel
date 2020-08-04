#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDir>
#include <QThread>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    killFlag(false),
    cameraThread(&MainWindow::cameraThreadFcn, this)
{
    std::unique_lock<std::mutex> lock(UiMtx);

    this->setFixedSize(422, 575);

    ui->setupUi(this);

    ui->sourceImageDisplay->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    ui->targetImageDisplay->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    connect(ui->TLx, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
    connect(ui->TLy, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
    connect(ui->TRx, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
    connect(ui->TRy, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
    connect(ui->BLx, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
    connect(ui->BLy, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
    connect(ui->BRx, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
    connect(ui->BRy, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::spinBox_changed);
}

MainWindow::~MainWindow()
{
    killFlag = true;
    cameraThread.join();

    delete ui;
}

QPixmap MainWindow::MatToPixmap(cv::Mat src)
{
    QImage::Format format=QImage::Format_Grayscale8;
    int bpp=src.channels();
    if(bpp==3)format=QImage::Format_RGB888;
    QImage img(src.cols,src.rows,format);
    uchar *sptr,*dptr;
    int linesize=src.cols*bpp;
    for(int y=0;y<src.rows;y++){
        sptr=src.ptr(y);
        dptr=img.scanLine(y);
        memcpy(dptr,sptr,linesize);
    }
    if(bpp==3)return QPixmap::fromImage(img.rgbSwapped());
    return QPixmap::fromImage(img);
}

void MainWindow::initSpinBoxValues(){
    ui->TRx->setValue(sourceImg.cols);

    ui->BLy->setValue(sourceImg.rows);

    ui->BRx->setValue(sourceImg.cols);
    ui->BRy->setValue(sourceImg.rows);
}

void MainWindow::updateSpinBoxProperties(){
    static bool fstScan = true;

    if(fstScan){
        initSpinBoxValues();
        fstScan = false;
    }

    ui->TLx->setRange(0, ui->TRx->value()-1);
    ui->TLy->setRange(0, ui->BLy->value()-1);

    ui->TRx->setRange(ui->TLx->value()+1, sourceImg.cols);
    ui->TRy->setRange(0, ui->BRy->value()-1);

    ui->BLx->setRange(0, ui->BRx->value()-1);
    ui->BLy->setRange(ui->TLy->value()+1, sourceImg.rows);

    ui->BRx->setRange(ui->BLx->value()+1, sourceImg.cols);
    ui->BRy->setRange(ui->TRy->value()+1, sourceImg.rows);
}

#include <iostream>
using namespace std;

void MainWindow::getNewFrame(){
    static bool fstScan = true;

    QDir d;
    std::string imgPath = d.absolutePath().toStdString() + "/board.jpg";
    sourceImg = cv::imread(imgPath, cv::IMREAD_COLOR);
    targetImg = cv::imread(imgPath, cv::IMREAD_COLOR);

    if(fstScan){
        updateSpinBoxProperties();
        fstScan = false;
    }

    cv::Point2f source[4] = {cv::Point(ui->TLx->value(),ui->TLy->value()),
                         cv::Point(ui->TRx->value(),ui->TRy->value()),
                         cv::Point(ui->BLx->value(),ui->BLy->value()),
                         cv::Point(ui->BRx->value(),ui->BRy->value())};

    cv::Point2f target[4] = {cv::Point(0,0),
                            cv::Point(sourceImg.cols,0),
                            cv::Point(0,sourceImg.rows),
                            cv::Point(sourceImg.cols,sourceImg.rows)};

    cv::Mat transformation = cv::getPerspectiveTransform(source,target);
    cv::warpPerspective(targetImg, targetImg, transformation, cv::Size(targetImg.cols, targetImg.rows));
}

void MainWindow::drawSkewPoints(){
    cv::circle(sourceImg, cv::Point(ui->TLx->value(), ui->TLy->value()), 20, cv::Scalar(0,0,255), -1);
    cv::circle(sourceImg, cv::Point(ui->TRx->value(), ui->TRy->value()), 20, cv::Scalar(0,0,255), -1);
    cv::circle(sourceImg, cv::Point(ui->BLx->value(), ui->BLy->value()), 20, cv::Scalar(0,0,255), -1);
    cv::circle(sourceImg, cv::Point(ui->BRx->value(), ui->BRy->value()), 20, cv::Scalar(0,0,255), -1);

    cv::circle(targetImg, cv::Point(0, 0), 20, cv::Scalar(0,0,255), -1);
    cv::circle(targetImg, cv::Point(0, targetImg.rows), 20, cv::Scalar(0,0,255), -1);
    cv::circle(targetImg, cv::Point(targetImg.cols, 0), 20, cv::Scalar(0,0,255), -1);
    cv::circle(targetImg, cv::Point(targetImg.cols, targetImg.rows), 20, cv::Scalar(0,0,255), -1);
}

void MainWindow::displayFrames(){
    QPixmap p = MatToPixmap(sourceImg);
    p = p.scaled(ui->sourceImageDisplay->width(), ui->sourceImageDisplay->height(), Qt::KeepAspectRatio);
    ui->sourceImageDisplay->setPixmap(p);

    p = MatToPixmap(targetImg);
    p = p.scaled(ui->targetImageDisplay->width(), ui->sourceImageDisplay->height(), Qt::KeepAspectRatio);
    ui->targetImageDisplay->setPixmap(p);
}

void MainWindow::cameraThreadFcn(){
    while(1){
        std::unique_lock<std::mutex> lock(UiMtx);
        getNewFrame();
        drawSkewPoints();
        displayFrames();
        lock.unlock();

        if(killFlag)
            return;

        QThread::msleep(40);
    }
}

void MainWindow::spinBox_changed(int unused){
    std::unique_lock<std::mutex> lock(UiMtx);
    updateSpinBoxProperties();
}

void MainWindow::on_pushButton_clicked()
{
    QDir d;
    QString fileName = QFileDialog::getSaveFileName(this,
        "Save config", d.absolutePath(), "Config files (*.conf)");
    if(fileName == ""){
        return;
    }

    QFile f(fileName);
    if(!f.open(QIODevice::WriteOnly)){
        QMessageBox msgBox;
        msgBox.setText("Couldn't open file for write.");
        msgBox.exec();
        return;
    }

    QTextStream stream(&f);
    stream << ui->TLx->value() << " " << ui->TLy->value() << " ";
    stream << ui->TRx->value() << " " << ui->TRy->value() << " ";
    stream << ui->BLx->value() << " " << ui->BLy->value() << " ";
    stream << ui->BRx->value() << " " << ui->BRy->value() << " ";

    f.close();
}
