#ifndef _THREADFUNCS_H_
#define _THREADFUNCS_H_

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "include/preProcessor.hpp"
#include "include/spatialPyr.hpp"
#include "include/temporalFilter.hpp"
#include "include/magnifier.hpp"

std::mutex readMutex;

extern int listSize;

cv::VideoCapture cap;
cv::Mat orginFrame;
cv::Mat floatFrame;

int curReaderIDX;
std::vector<cv::Mat> splitFrame;
std::vector<cv::Mat> yChannelList;
std::vector<cv::Mat> iChannelList;
std::vector<cv::Mat> qChannelList;
std::vector<cv::Mat> vidFFTList;


std::vector<cv::Mat> fftList;

preProcessor preProc;
magnifier magProc;

#endif
