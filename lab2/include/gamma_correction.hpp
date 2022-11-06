#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    void gammaCurveMain(cv::Mat &image);

    auto applyGammaCorrection(const cv::Mat &image, double gamma);
}
