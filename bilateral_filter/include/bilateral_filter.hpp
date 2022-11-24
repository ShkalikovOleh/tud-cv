#pragma once

#include <functional>
#include <opencv2/opencv.hpp>

namespace tud
{
    typedef std::function<float(float)> wfunc;

    cv::Mat bilateral_filter(const cv::Mat &image, uint n, wfunc ds_func, wfunc dc_func);
}