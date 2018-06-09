#ifndef _SPATIALPYR_HPP_
#define _SPATIALPYR_HPP_
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <vector>

class spatialPyr {
 public:
        spatialPyr(cv::Size srcSize, int numLevel);
        void pyrBuild();
        void pyrRecon();
        void buildLevel(cv::Mat &srcImg, int filterIDX, cv::Mat &dstImg);
        void reconLevel(cv::Mat &srcImg, int filterIDX, cv::Mat &dstImg);
        void getFilters(std::vector<double> const &rVals, int orientations, int twidth);
        void octaveFilter();
        std::vector<cv::Mat> pyrFilters;

 private:
        int pyrLevel;
        int srcHeight;
        int srcWidth;
        cv::Mat meshgridX;
        cv::Mat meshgridY;
        cv::Mat magnitudeGrid;
        cv::Mat angleGrid;
        cv::Mat hiMask;
        cv::Mat loMask;
        cv::Mat angleMask;
        void meshgrid(int width, int height);
        void getPolarGrid();
        void getRadialMaskPair(double r, double twidth);
        void getAngleMask(int b, int orientations);
};

#endif
