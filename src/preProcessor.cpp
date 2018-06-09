#include "../include/preProcessor.hpp"

using namespace std;
using namespace cv;

preProcessor::preProcessor(Mat &srcImg) {
    // Pre allocation use vector?

}

void preProcessor::bgr2ntsc(Mat &bgrImg, Mat yiqChannels[3]) {
    /* convert bgr image to ntsc(YIQ) image
        Y     0.299  0.587  0.114   R
        I  =  0.596 -0.274 -0.322   G
        Q     0.211 -0.523  0.312   B
        src and dst needs to be signed (float) value, otherwise cause serious color distortion.
    */
    cv::split(bgrImg, splitBuffer);
    yiqChannels[0] = 0.299 * splitBuffer[2] + 0.588 * splitBuffer[1] + 0.114 * splitBuffer[0];
    yiqChannels[1] = 0.596 * splitBuffer[2] - 0.274 * splitBuffer[1] - 0.322 * splitBuffer[0];
    yiqChannels[2] = 0.211 * splitBuffer[2] - 0.523 * splitBuffer[1] + 0.312 * splitBuffer[0];

    // cv::merge(yiqChannels, 3, ntscImg);
}


void preProcessor::ntsc2bgr(Mat &ntscImg, Mat bgrChannels[3]) {
    /* convert ntsc(YIQ) image to bgr image
     R   1.000  0.956    0.621     Y
     G = 1.000  -0.272   -0.647 *  I
     B   1.000  -1.105   1.702     Q
     src and dst needs to be signed(float) value, otherwise cause serious color distortion.
    */
    cv::split(ntscImg, splitBuffer);
    bgrChannels[2] = splitBuffer[0] + 0.956 * splitBuffer[1] + 0.621 * splitBuffer[2];
    bgrChannels[1] = splitBuffer[0] - 0.272 * splitBuffer[1] - 0.647 * splitBuffer[2];
    bgrChannels[0] = splitBuffer[0] - 1.105 * splitBuffer[1] + 1.702 * splitBuffer[2];

    // cv::merge(bgrChannels, 3, bgrImg);
}

void preProcessor::ntsc2bgr(Mat yiqChannels[3], Mat bgrChannels[3], Mat bgrImg) {
    bgrChannels[2] = yiqChannels[0] + 0.956 * yiqChannels[1] + 0.621 * yiqChannels[2];
    bgrChannels[1] = yiqChannels[0] - 0.272 * yiqChannels[1] - 0.647 * yiqChannels[2];
    bgrChannels[0] = yiqChannels[0] - 1.105 * yiqChannels[1] + 1.702 * yiqChannels[2];

    cv::merge(bgrChannels, 3, bgrImg);
}


void preProcessor::vid2DFT(vector<Mat> &srcImg, Mat &dstImg) {
    // complexImg[0] = srcImg;
    merge(srcImg, dstImg);
    dft(dstImg, dstImg, DFT_COMPLEX_OUTPUT);
}

// Convert Freq back to spatial
void preProcessor::DFT2vid(Mat &srcImg, Mat &dstImg) {
    idft(srcImg, dstImg, DFT_REAL_OUTPUT|DFT_SCALE);
}
