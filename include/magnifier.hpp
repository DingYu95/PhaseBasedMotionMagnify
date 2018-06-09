#ifndef _MAGNIFIER_HPP_
#define _MAGNIFIER_HPP_

#include <iostream>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "../include/spatialPyr.hpp"
#include "../include/temporalFilter.hpp"


class magnifier {
 public:
    magnifier(cv::Size2i imgSize, int numLevel, int stack);
    bool fillStack;

    spatialPyr spatialFilter;
    firFilter temporalFilter;
    int stackSize;
    float magPhase = 3.f;

    cv::Mat magnifiedLumaFFT;  // shared between different levels

    // TODO(me): [optimization] create lists of buffers, magnify levels concurrently
    // Each level should have its own copy of following variable
    cv::Mat pyrRef;      // Select a frame as reference
    std::vector<cv::Mat> deltaStacks;  // Stack of images reshaped as colums for DFT in time axis
    cv::Mat deltaStackDFT;

    // buffer variables updated in magify()
    cv::Mat curLevelFrame;  // Also filterResponse, originalLevel, lowpassFrame
    cv::Mat pyrCurrent[2];  // Also Mat expForm[2];
    cv::Mat tempTransformOut;
    cv::Mat deltaCurrent;

    cv::Mat intModMat;
    cv::Mat floatModMat;

    void magnifyLevel(cv::Mat &srcImg, int levelIDX, int frameIDX);
    void maginify(cv::Mat &srcImg, int frameIDX);
    void tempFiltering(cv::Mat &srcImg, int levelIDX, int frameIDX, cv::Mat &dstImg);
    void matMod(cv::Mat &srcImg, cv::Mat &dstImg);
};

#endif
