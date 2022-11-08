#pragma once
#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    auto applyGammaCorrection(const cv::Mat &image, double gamma);

    void gammaCurveMain(cv::Mat &image);
}
