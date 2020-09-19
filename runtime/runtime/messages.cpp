#include "messages.h"

#include <iostream>
#include <mutex>

#include "thirdparty/termcolor/include/termcolor/termcolor.hpp"

namespace  {
    mutex outMtx;
}

/* external functions */
void normal(string msg);
void info(string msg);
void success(string msg);
void warn(string msg);
void error(string msg);

/* external functions definitions */
void normal(string msg){
    unique_lock<mutex> lock(outMtx);
    cout << msg;
}

void info(string msg){
    unique_lock<mutex> lock(outMtx);
    cout << termcolor::cyan << msg << termcolor::reset;
}

void success(string msg){
    unique_lock<mutex> lock(outMtx);
    cout << termcolor::green << msg << termcolor::reset;
}

void warn(string msg){
    unique_lock<mutex> lock(outMtx);
    cout << termcolor::yellow << msg << termcolor::reset;
}

void error(string msg){
    unique_lock<mutex> lock(outMtx);
    cout << termcolor::red << msg << termcolor::reset;
}
