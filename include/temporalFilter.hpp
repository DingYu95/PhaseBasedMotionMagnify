#ifndef _TEMPORALFILTER_HPP_
#define _TEMPORALFILTER_HPP_

#include <iostream>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

class temporalFilter {
 public:
    temporalFilter();
    void filtering(cv::Mat srcImg);
};

// TODO(me): [feature] use inherence
class firFilter {
 public:
    firFilter();
    void filtering();
    void genFirBandpass(int length, int shift);
    cv::Mat bandpassDFT;
    int imgTotal;
 private:
    double fl;  // lowpass
    double fh;  // highpass
};

#endif
