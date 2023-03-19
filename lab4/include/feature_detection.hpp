#pragma once

#include <opencv2/opencv.hpp>

#include <tuple>

namespace tud::cvlabs
{
    using StructureTensorComponents = std::tuple<cv::Mat, cv::Mat, cv::Mat>;

    StructureTensorComponents computeStructureTensor(const cv::Mat &image);

    cv::Mat computeLowerEigVals(const cv::Mat &image);

    cv::Mat computeR(const cv::Mat &image, float k = 0.05);

    cv::Mat detectCornersHariss(const cv::Mat &image, float threshold, float k = 0.05);

    cv::Mat nms(const cv::Mat &image);
}