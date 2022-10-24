#pragma once
#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    cv::Mat localMaxNaive(const cv::Mat &image, const cv::Mat &neigh_mask);

    cv::Mat localMaxNaiveParallel(const cv::Mat &image, const cv::Mat &neigh_mask);

    cv::Mat localMaxDilate(const cv::Mat &image, const cv::Mat &neigh_mask);

    cv::Mat maxPlateaus(const cv::Mat &image, const cv::Mat &neigh_mask);
}