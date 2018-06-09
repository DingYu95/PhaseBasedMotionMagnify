#ifndef _PREPROCESSOR_HPP_
#define _PREPROCESSOR_HPP_

#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

class preProcessor {
 public:
    preProcessor(cv::Mat &src);
    void vid2DFT(std::vector<cv::Mat> &srcImg, cv::Mat &dstImg);
    void DFT2vid(cv::Mat &srcImg, cv::Mat &dstImg);
    // TODO(me): [optimization] handle split arrays better for multithreading
    void bgr2ntsc(cv::Mat &bgrImg, cv::Mat yiqChannels[3]);
    void ntsc2bgr(cv::Mat &ntscImg, cv::Mat bgrChannels[3]);
    void ntsc2bgr(cv::Mat yiqChannels[3], cv::Mat bgrChannels[3], cv::Mat bgrImg);
 private:
    cv::Size imgSize;
    // Unnecessary for regular frame size like 480, 640, 720
    // cv::Size dftSize; getOptimalDFTSize();
    cv::Mat dftImg;

    cv::Mat splitBuffer[3];
};

#endif
