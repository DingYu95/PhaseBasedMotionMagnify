#include "../include/magnifier.hpp"

using namespace std;
using namespace cv;


magnifier::magnifier(Size2i imgSize, int numLevel, int stack):
    spatialFilter(imgSize, numLevel), stackSize(stack) {
    fillStack = true;
    spatialFilter.octaveFilter();
    temporalFilter.imgTotal = imgSize.width * imgSize.height;
    temporalFilter.genFirBandpass(stackSize, 0);
    magnifiedLumaFFT = Mat::zeros(imgSize, CV_32FC2);
    for (int i = 0; i < spatialFilter.pyrFilters.size(); i++) {
        deltaStacks.push_back(Mat(temporalFilter.imgTotal, stackSize, CV_32FC1));
    }
    intModMat = Mat::zeros(imgSize, CV_8UC2);
    floatModMat = Mat::zeros(imgSize, CV_32FC2);
}

void magnifier::tempFiltering(Mat &srcImg, int levelIDX, int frameIDX, Mat &dstImg) {
    Mat deltaStack = deltaStacks[levelIDX];
    // srcImg = srcImg.reshape(0, srcImg.total());
    // srcImg.copyTo(deltaStack.col(frameIDX));

    dft(deltaStack, deltaStackDFT, DFT_COMPLEX_OUTPUT|DFT_ROWS);  // |DFT_SCALE
    // Mat bandpass = temporalFilter.bandpassDFT.clone();
    mulSpectrums(deltaStackDFT, temporalFilter.bandpassDFT, deltaStackDFT, 0);
    idft(deltaStackDFT, deltaStack, DFT_REAL_OUTPUT|DFT_ROWS|DFT_SCALE);

    dstImg = deltaStack.col(frameIDX).clone().reshape(0, curLevelFrame.rows);  // phaseOfFrame
}


void magnifier::magnifyLevel(Mat &srcImg, int levelIDX, int frameIDX) {
        spatialFilter.buildLevel(srcImg, levelIDX, curLevelFrame);  // filterResponse
        split(curLevelFrame, pyrCurrent);  // filterResponse
        phase(pyrCurrent[0], pyrCurrent[1], pyrCurrent[1]);

        // TODO(me): [feature] better reference frame
        if (frameIDX % stackSize == 0) {
            pyrRef = pyrCurrent[1].clone();
        }
        deltaCurrent = pyrCurrent[1] - pyrRef;

        // Modulo
        matMod(deltaCurrent, deltaCurrent);

        // insert to stack of images
        deltaCurrent = deltaCurrent.reshape(0, deltaCurrent.total());
        deltaCurrent.copyTo(deltaStacks[levelIDX].col(frameIDX));

        // return immediately after copying image to stack
        if (fillStack) {
            if (frameIDX < stackSize -1) {
                deltaCurrent = deltaCurrent.reshape(0, curLevelFrame.rows);
                return;
            }
            fillStack = false;
        }

        // Apply temporal filter
        tempFiltering(deltaCurrent, levelIDX, frameIDX, deltaCurrent);  // phase of response after temporal filtering

        // Magnify
        deltaCurrent.convertTo(deltaCurrent, -1, magPhase);

        // Use polarToCart to get exp(i*a) = cos(a) + i*sin(a)
        polarToCart(Mat(), deltaCurrent, pyrCurrent[0], pyrCurrent[1]);  // expForm[2]
        merge(pyrCurrent, 2, tempTransformOut);
        // TODO(me): [feature] add attenuateOtherFreq as, as in matlab code
        multiply(tempTransformOut, curLevelFrame, tempTransformOut);  // filterResponse as originalLevel

        spatialFilter.reconLevel(tempTransformOut, levelIDX, curLevelFrame);
        add(magnifiedLumaFFT, curLevelFrame, magnifiedLumaFFT);
}


// @ srcImg: fft2 of input video frame
void magnifier::maginify(Mat &srcImg, int frameIDX) {
    // TODO(me): [optimization] enable multithreading, magnify levels concurrently
    for (int i = 0; i < spatialFilter.pyrFilters.size() - 1; i++) {
        magnifyLevel(srcImg, i, frameIDX);
    }

    // Add unmolested lowpass residual, pow(loPass, 2) pre-computed in creatation
    multiply(srcImg, spatialFilter.pyrFilters.back(), curLevelFrame);  // lowpassFrame
    add(magnifiedLumaFFT, curLevelFrame, magnifiedLumaFFT);
}

void magnifier::matMod(Mat &srcImg, Mat &dstImg) {
    srcImg.convertTo(intModMat, CV_8UC2, M_1_PI/2);  // int(a/(2*PI))
    intModMat.convertTo(floatModMat, CV_32FC2, 2*M_PI);
    subtract(srcImg, floatModMat, dstImg);
}
