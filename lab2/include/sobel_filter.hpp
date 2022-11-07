#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    cv::Mat applySobel(const cv::Mat &image, uint ddepth, cv::BorderTypes borderType);

    void sobelMain(const cv::Mat &image);
}