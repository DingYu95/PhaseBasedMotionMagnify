#include "../include/temporalFilter.hpp"

using namespace std;
using namespace cv;

firFilter::firFilter() {
    fh = .7f;
    fl = .4f;
}

void firFilter::genFirBandpass(int order, int shift = 0) {
    if (order%2) {
        cout<< "need even filter order"<< endl;
    }
    Mat bandpassFIR = Mat::zeros(1, order, CV_32FC1);
    float w;
    float hw;  // Hamming window

    // An appriximation of fir1(order, [low, high]) in matlab
    for (int i=0; i < order; i++) {
        w = sin(2*M_PI*fh/2*(i-order/2))/(M_PI*(i-order/2)) - sin(2*M_PI*fl/2*(i-order/2))/(M_PI*(i-order/2));
        hw = 0.54 - 0.46 * cos(2*M_PI*i / order);
        bandpassFIR.at<float> (0, i)= w * hw;
    }
    bandpassFIR.at<float> (0, order/2) = fh - fl;

    Mat tmpShift = Mat(bandpassFIR.size(), bandpassFIR.type());
    bandpassFIR.colRange(0, order/2-1).copyTo(tmpShift.colRange(order/2, order-1));
    bandpassFIR.colRange(order/2, order-1).copyTo(tmpShift.colRange(0, order/2-1));

    dft(tmpShift, bandpassDFT, DFT_COMPLEX_OUTPUT);

    // As image stack are arranged as (width * height) * stack_size
    repeat(bandpassDFT, imgTotal, 1, bandpassDFT);
}
