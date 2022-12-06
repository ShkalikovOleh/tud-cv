#include "integral.hpp"

namespace tud::cvlabs
{
    int getResultType(int depth, int nchannels)
    {
        switch (depth)
        {
        case CV_8U:
            return CV_32SC(nchannels);
        case CV_32F:
            return CV_32FC(nchannels);
        default:
            return CV_64FC(nchannels);
        }
    }

    cv::Mat integrate(const cv::Mat &image)
    {
        int r = image.rows, c = image.cols, ch = image.channels();
        auto type = getResultType(image.depth(), ch);

        cv::Mat result(r + 1, c + 1, type);
        image.convertTo(result(cv::Rect(1, 1, c, r)), type);

        for (int i = 1; i < r + 1; ++i)
        {
            result.row(i) = result.row(i) + result.row(i - 1);
        }

        for (int j = 1; j < c + 1; ++j)
        {
            result.col(j) += result.col(j - 1);
        }

        return result;
    }
}