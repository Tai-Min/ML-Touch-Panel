#QT -= gui
QT       += core gui multimedia widgets
CONFIG += c++11 console
#CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        audioprocessor.cpp \
        eventdetectorthread.cpp \
        main.cpp \
        mainwindow.cpp \
        messages.cpp \
        networkthread.cpp \
        pointertrackerthread.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    audioprocessor.h \
    eventdetectorthread.h \
    mainwindow.h \
    messages.h \
    networkthread.h \
    pointertrackerthread.h \
    thirdparty/Signal-Utilities-for-Arduino/SignalFlow/Analog/LowPassFilter/LowPassFilter.h \
    thirdparty/json/single_include/nlohmann/json.hpp \
    thirdparty/termcolor/include/termcolor/termcolor.hpp

INCLUDEPATH += $$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/opencv/include \
               $$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/inference_engine/include \
               $$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/deployment_tools/ngraph/include

LIBS += -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/opencv/lib \
        -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/opencv/bin \
        -lopencv_core440 \
        -lopencv_highgui440 \
        -lopencv_imgcodecs440 \
        -lopencv_imgproc440 \
        -lopencv_videoio440 \
        -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/inference_engine/lib/intel64/Release \
        -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/inference_engine/bin/intel64/Release \
        -linference_engine \
        -linference_engine_preproc \
        -linference_engine_legacy \
        -linference_engine_transformations \
        -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/inference_engine/external/tbb/lib \
        -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/inference_engine/external/tbb/bin \
        -ltbb \
        -L$$PWD/../../../Programy/Programowanie/ML/openvino/openvino_2020.4.287/deployment_tools/ngraph/lib \
        -lngraph \
        -luser32
        #-lngraph_test_util \
        #-lie_backend \
        #-linterpreter_backend \
        #-lonnx_importer

FORMS += \
    mainwindow.ui
