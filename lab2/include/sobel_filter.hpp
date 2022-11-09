#pragma once
#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    cv::Mat applySobel(const cv::Mat &image, uint ddepth = CV_8U);

    void sobelMain(const cv::Mat &image);
}