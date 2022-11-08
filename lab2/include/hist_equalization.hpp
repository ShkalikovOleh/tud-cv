#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    cv::Mat applyEqualization(const cv::Mat &image, cv::Mat &H);

    cv::Mat calculateHistogram(const cv::Mat &image);
}