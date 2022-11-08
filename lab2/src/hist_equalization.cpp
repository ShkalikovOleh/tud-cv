#include <numeric>

#include "hist_equalization.hpp"

namespace tud::cvlabs
{
    cv::Mat applyEqualization(const cv::Mat &image, cv::Mat &H)
    {
        std::vector<double> cdf;
        std::inclusive_scan(H.begin<double>(), H.end<double>(),
                            std::back_inserter(cdf), std::plus<double>{});

        std::vector<uchar> lut;
        std::transform(cdf.begin(), cdf.end(), std::back_inserter(lut),
                       [](double cdf)
                       {
                           return cv::saturate_cast<uchar>(cdf * 255);
                       });

        cv::Mat lutMat(lut);

        cv::Mat result;
        cv::LUT(image, lutMat, result);

        return result;
    }

    cv::Mat calculateHistogramForChannel(const cv::Mat &channel)
    {
        auto range = cv::Range(0, 256);
        cv::Mat result = cv::Mat::zeros(1, 256, CV_64F);

        float norm = std::log(channel.rows * channel.cols);
        cv::parallel_for_(range, [&channel, &result, norm](cv::Range rng)
                          {
                            for (int k = rng.start; k < rng.end; ++k)
                            {
                                int n = cv::countNonZero(channel == k);
                                result.at<double>(0, k) = std::exp(std::log(n) - norm);
                            } });

        return result;
    }

    cv::Mat calculateHistogram(const cv::Mat &image)
    {
        std::vector<cv::Mat> imgChannels, resChannels;
        cv::split(image, imgChannels);

        std::transform(imgChannels.begin(), imgChannels.end(), std::back_inserter(resChannels),
                       [](const cv::Mat &channel)
                       {
                           return calculateHistogramForChannel(channel);
                       });

        cv::Mat result;
        cv::merge(resChannels, result);

        return result;
    }
}