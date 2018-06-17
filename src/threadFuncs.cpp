#include "../include/threadFuncs.hpp"

using namespace std;
using namespace cv;

// read from camera, covert color, image to dft
void readFrame() {
    while (true) {
        cap >> orginFrame;
        orginFrame.convertTo(floatFrame, CV_64FC3);
        split(floatFrame, splitFrame);
        {
        lock_guard<mutex> readLock(readMutex);
        yChannelList[curReaderIDX] = 0.299 * splitFrame[2] + 0.588 * splitFrame[1] + 0.114 * splitFrame[0];
        iChannelList[curReaderIDX] = 0.596 * splitFrame[2] - 0.274 * splitFrame[1] - 0.322 * splitFrame[0];
        qChannelList[curReaderIDX] = 0.211 * splitFrame[2] - 0.523 * splitFrame[1] + 0.312 * splitFrame[0];
        dft(yChannelList[curReaderIDX], vidFFTList[curReaderIDX], DFT_COMPLEX_OUTPUT|DFT_SCALE);
        curReaderIDX++;
        curReaderIDX %= listSize;
        }
        // this_thread::sleep_for(chrono::duration)
    }
}
