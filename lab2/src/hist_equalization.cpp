#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "hist_equalization.hpp"

namespace tud::cvlabs
{
    cv::Mat applyEqualization(const cv::Mat &image, cv::Mat &hist)
    {
        std::vector<double> cdf;
        std::inclusive_scan(hist.begin<double>(), hist.end<double>(),
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

    void showHist(const cv::Mat &hist, const cv::String &wndName)
    {
        std::vector<uchar> y;

        double max = *std::max_element(hist.begin<double>(), hist.end<double>());
        double norm = 255 / max;

        std::transform(hist.begin<double>(), hist.end<double>(), std::back_inserter(y),
                       [norm](double h)
                       {
                           return cv::saturate_cast<uchar>(h * norm);
                       });

        cv::Mat result = cv::Mat::zeros(256, 256, CV_8U);
        for (int i = 0; i < 256; ++i)
        {
            cv::rectangle(result, cv::Rect(i, 255 - y[i], 1, y[i]), 255);
        }

        imshow(wndName, result);
    }

    void histEqualMain(const cv::Mat &image)
    {
        cv::Mat grayImg;
        cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

        auto origHist = calculateHistogram(grayImg);
        auto result = applyEqualization(grayImg, origHist);
        auto resHist = calculateHistogram(result);

        showHist(origHist, "Original histogram");
        showHist(resHist, "Result histogram");
        imshow("Our Hist Equalization", result);
        imshow("Original Image", grayImg);

        cv::Mat reference;
        cv::equalizeHist(grayImg, reference);
        imshow("Reference Hist Equalization", reference);

        cv::waitKey();
    }
}