#include "../include/spatialPyr.hpp"

using namespace std;
using namespace cv;

spatialPyr::spatialPyr(Size srcSize, int numLevel):pyrLevel(numLevel) {
    srcHeight = srcSize.height;
    srcWidth = srcSize.width;
    meshgridX = Mat::zeros(srcHeight, srcWidth, CV_32FC1);
    meshgridY = Mat::zeros(srcHeight, srcWidth, CV_32FC1);
    for (int i = 0; i < 14; i ++) {
        pyrFilters.push_back(Mat(srcSize, CV_32FC2, Scalar::all(0)));
    }
    // meshgrid(srcHeight, srcWidth);
    // getPolarGrid();
}

// shift(flip) filter, instead of fftshift(src)
void spatialPyr::buildLevel(Mat &srcImg, int filterIDX, Mat &dstImg) {
    // srcImg, pyrFilter are complex mat(2-channel)
    mulSpectrums(srcImg, pyrFilters[filterIDX], dstImg, 0);
    idft(dstImg, dstImg, DFT_COMPLEX_OUTPUT|DFT_SCALE);
}

void spatialPyr::reconLevel(Mat &srcImg, int filterIDX, Mat &dstImg) {
    // srcImg is complex mat(2-channel)
    dft(srcImg, dstImg, DFT_COMPLEX_OUTPUT|DFT_SCALE);
    multiply(dstImg, pyrFilters[filterIDX], dstImg, 2.0);
}

void spatialPyr::meshgrid(int height, int width) {
    // Possible improvement use cv::repeat()
    vector<float> rowX(width);
    vector<float> colY(height);
    for (int i = 0; i < width; i++) {
        rowX[i] = i;
    }
    for (int j = 0; j < height; j ++) {
        colY[j] = j;
    }
    repeat(Mat(rowX).reshape(1, 1), height, 1, meshgridX);
    repeat(Mat(colY).reshape(1, height), 1, width, meshgridY);

    // int16_t* meshgridX_ptr = (int16_t*) meshgridX.data;
    // int16_t* meshgridY_ptr = (int16_t*) meshgridY.data;
    // for (int i = 0; i < height; i++) {
    //     for (int j = 0; j < height; j++) {
    //         meshgridX_ptr[i*width + j] = j;
    //         meshgridY_ptr[i*width + j] = i;
    //     }
    // }
}

void spatialPyr::getPolarGrid() {
    // Trival difference to origin matlab code:
    // 1. not bounded angleGrid to [0, 2*pi] or [-pi, pi]
    subtract(meshgridX, (srcWidth-0.0)/2 * Mat::ones(srcHeight, srcWidth, CV_32FC1), meshgridX);
    meshgridX.convertTo(meshgridX, -1, 2.0/srcWidth);
    subtract(meshgridY, (srcHeight-0.0)/2 * Mat::ones(srcHeight, srcWidth, CV_32FC1), meshgridY);
    meshgridY.convertTo(meshgridY, -1, 2.0/srcWidth);
    cartToPolar(meshgridX, meshgridY, magnitudeGrid, angleGrid);
    magnitudeGrid.at<float>(srcHeight/2, srcWidth/2) = magnitudeGrid.at<float>(srcHeight/2, srcWidth/2-1);
}

void spatialPyr::getRadialMaskPair(double r, double twidth) {
    Mat tmpBase2 = 2.0 * Mat::ones(srcHeight, srcWidth, CV_32FC1);  // Used for convert log to log2
    log(tmpBase2, tmpBase2);

    Mat magnitudeGrid_copy;
    magnitudeGrid.convertTo(magnitudeGrid_copy, -1, 1/r);  // log(magnitudeGrid) - log(r) = log(magnitudeGrid/r)
    log(magnitudeGrid_copy, hiMask);
    divide(hiMask, tmpBase2, hiMask);                      // log_b(a) = log_c(a) / log_c(b)

    // Clip
    hiMask.setTo(-twidth, hiMask < -twidth);
    hiMask.setTo(0, hiMask > 0);

    hiMask.convertTo(hiMask, -1, M_PI/(2*twidth));
    polarToCart(Mat(), hiMask, hiMask, loMask);
    hiMask = abs(hiMask);
    loMask = abs(loMask);
}

void spatialPyr::getAngleMask(int b, int orientations) {
    int order = orientations - 1;
    int16_t order_factorial1 = 1;
    for (int i = 1; i < order+1; i++) {
        order_factorial1 *= i;
    }
    int order_factorial2 = order_factorial1;
    for (int i = order+1; i < 2*order+1; i++) {
        order_factorial2 *= i;
    }

    double scale_const = pow(2, (2 * order)) * pow(order_factorial1, 2) / (orientations * order_factorial2);

    // Modulo of angleGrid
    Mat angleGrid_copy(angleGrid.size(), angleGrid.type());
    // float* angleGrid_copy_ptr = (float*) angleGrid_copy.data;
    // float* angleGrid_ptr = (float*) angleGrid.data;
    for (int i = 0; i < srcHeight; i++) {
        for (int j = 0; j < srcWidth; j++) {
            angleGrid_copy.data[i*srcWidth+j] =
                fmod(M_PI+angleGrid.data[i*srcWidth+j]-M_PI*(b-1)/orientations, 2*M_PI) - M_PI;
        }
    }

    Mat falloff;
    threshold(abs(angleGrid_copy), falloff, M_PI_2, 1, THRESH_BINARY_INV);

    // Get cos(angleGrid)
    Mat tmp = Mat::ones(angleGrid.size(), angleGrid.type());
    polarToCart(Mat(), angleGrid_copy, angleGrid_copy, tmp);

    cv::pow(angleGrid_copy, order, angleGrid_copy);

    multiply(angleGrid_copy, falloff, angleMask, 2*sqrt(scale_const), -1);
}

void spatialPyr::getFilters(vector<double> const &rVals, int orientations, int twidth = 1) {
    vector<Mat> filters((rVals.size() -1) * orientations+2);

    int count = 0;
    getRadialMaskPair(rVals[0], twidth);
    filters[count] = hiMask.clone();
    count++;
    Mat lomaskPrev = loMask.clone();
    for (int i = 1; i < rVals.size(); i++) {
        getRadialMaskPair(rVals[i], twidth);
        Mat radMask;
        multiply(hiMask, lomaskPrev, radMask);
        for (int j = 1; j < orientations + 1; j++) {
            getAngleMask(j, orientations);
            multiply(radMask, angleMask/2, filters[count]);
            // cout<<filters[count]<<endl;
            count++;
        }
        lomaskPrev = loMask.clone();
    }
    filters[count] = loMask.clone();

    // Check and reset nan numbers
    // TODO(me): fix nans more properly
    for (int i = 0; i < filters.size() - 1; i++) {
        Mat nanMask = Mat(filters[i] != filters[i]);
        if (countNonZero(nanMask) == 0)
            continue;
        filters[i].setTo(cv::Scalar(0, 0), nanMask);
    }
    // TODO(me): [optimization] use mask to denote non-zero, as in matlab code

    // FFT_shift filter
    // https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
    int cx = filters[count].cols/2;
    int cy = filters[count].rows/2;
    for (int i = 0; i < count - 1; i++) {
        // Create a ROI per quadrant
        Mat q0(filters[i], Rect(0, 0, cx, cy));    // Top-Left
        Mat q1(filters[i], Rect(cx, 0, cx, cy));   // Top-Right
        Mat q2(filters[i], Rect(0, cy, cx, cy));   // Bottom-Left
        Mat q3(filters[i], Rect(cx, cy, cx, cy));  // Bottom-Right
        Mat tmpQuad;                               // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmpQuad);
        q3.copyTo(q0);
        tmpQuad.copyTo(q3);
        q1.copyTo(tmpQuad);                        // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmpQuad.copyTo(q2);

        // Make filters 2-channel, compatible with complex Mat[2]
        Mat tmp[2] = {filters[i], filters[i]};
        merge(tmp, 2, pyrFilters[i]);
    }

    // pre-compute pow(loPass, 2)
    Mat loMaskPow2;
    pow(filters.back(), 2, loMaskPow2);
    Mat tmp[2] = {loMaskPow2, loMaskPow2};
    merge(tmp, 2, pyrFilters.back());

    // de-allocate intermediate results, only keep filters
    meshgridX.release();
    meshgridY.release();
    magnitudeGrid.release();
    angleGrid.release();
    hiMask.release();
    loMask.release();
    angleMask.release();
}

void spatialPyr::octaveFilter() {
    meshgrid(srcHeight, srcWidth);
    getPolarGrid();
    vector<double> rVals(pyrLevel);
    for (int i = 0; i < pyrLevel; i++) {
        rVals[i] = pow(2, -i);
    }
    getFilters(rVals, 4);
}
