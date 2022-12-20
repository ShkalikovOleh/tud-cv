#include <opencv2/opencv.hpp>

#include <vector>
#include <functional>

namespace tud::cvlabs
{
    using cfunc = std::function<std::vector<float>(int, int, cv::Vec3b)>;
    using ccfunc = std::function<std::vector<float>(int, int, cv::Vec3b, int, int, cv::Vec3b)>;

    cv::Mat classify(const cv::Mat &image, const cfunc &c, const ccfunc &cc);
}
