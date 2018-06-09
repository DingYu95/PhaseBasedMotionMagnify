#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "include/preProcessor.hpp"
#include "include/spatialPyr.hpp"
#include "include/temporalFilter.hpp"
#include "include/magnifier.hpp"

#define CPU_MODE 1
#define GPU_MODE 0

using namespace std;
using namespace cv;

int main() {
    VideoCapture cap("./face.mp4");  // "./face.mp4"

    int imgWidth;
    int imgHeight;
    int stackSize = 128;
    if (cap.isOpened() == false) {
        cout << "Cannot open the video camera" << endl;
        cin.get();
        return -1;
    }
    imgWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    imgHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    Mat bgrImg(imgHeight, imgWidth, CV_8UC3);
    Mat bgrFloat(imgHeight, imgWidth, CV_32FC3);

    Mat vidDFT(imgHeight, imgWidth, CV_32FC2);
    Mat magDFT(imgHeight, imgWidth, CV_32FC1);

    Mat yiqChannels[3];
    Mat bgrChannels[3];
    Mat magImg(imgHeight, imgWidth, CV_32FC3);
    Mat bgrResult(imgHeight, imgWidth, CV_8UC3);
    vector<Mat> complexImg = {Mat::zeros(bgrImg.size(), CV_32FC1), Mat::zeros(bgrImg.size(), CV_32FC1)};
    preProcessor preP(bgrFloat);

    int stackMax = 32;
    int frameIdx = 0;
    magnifier mag(Size2i(imgWidth, imgHeight), 4, stackMax);
    while (cap.isOpened()) {
        cap >> bgrImg;
        imshow("origin", bgrImg);
        bgrImg.convertTo(bgrFloat, CV_32FC3);

        preP.bgr2ntsc(bgrFloat, yiqChannels);
        complexImg[0] = yiqChannels[0];

        preP.vid2DFT(complexImg, vidDFT);
        mag.maginify(vidDFT, frameIdx);
        frameIdx++;
        frameIdx %= stackMax;
        preP.DFT2vid(mag.magnifiedLumaFFT, magDFT);
        mag.magnifiedLumaFFT.setTo(0);
        yiqChannels[0] = magDFT;
        preP.ntsc2bgr(yiqChannels, bgrChannels, magImg);
        magImg.convertTo(bgrResult, CV_8UC3);
        imshow("mag", bgrResult);
        if (27 == waitKey(10)%256) break;
    }

    return 0;
}