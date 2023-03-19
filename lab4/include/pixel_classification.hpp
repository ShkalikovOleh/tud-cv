#include <opencv2/opencv.hpp>

#include <vector>
#include <functional>

namespace tud::cvlabs
{
    // returns vector where i-th position is the cost of assigning class with label i
    using cfunc = std::function<std::vector<float>(int, int, cv::Vec3b)>;

    // cc func returns on i-th position cost for assigning for current pixel class label i
    // given neihbours position, class and color
    using ccfunc = std::function<std::vector<float>(int, int, cv::Vec3b, int, int, uchar, cv::Vec3b)>;

    cv::Mat classify(const cv::Mat &image, const cfunc &c, const ccfunc &cc);
}
