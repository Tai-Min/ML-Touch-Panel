#-------------------------------------------------
#
# Project created by QtCreator 2020-08-04T01:14:20
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = configurator
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        mainwindow.h

FORMS += \
        mainwindow.ui

INCLUDEPATH += $$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/opencv/include
LIBS += -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/opencv/lib \
        -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/opencv/bin \
        -lopencv_core440 \
        -lopencv_highgui440 \
        -lopencv_imgcodecs440 \
        -lopencv_imgproc440 \
        -lopencv_videoio440

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
