#pragma once

#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    cv::Mat medianFilter(const cv::Mat &image, int kw, int kh,
                         cv::BorderTypes borderType = cv::BorderTypes::BORDER_DEFAULT);
}