#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    void sobelKernel(const cv::Mat &image, cv::Mat &res, int x, int y, uint ddepth)
    {
        float value = 0;
        float neighValue;

        neighValue = image.ptr(x - 1)[y - 1];
        value -= neighValue;

        neighValue = image.ptr(x)[y - 1];
        value -= (neighValue + neighValue);

        neighValue = image.ptr(x + 1)[y - 1];
        value -= neighValue;

        neighValue = image.ptr(x - 1)[y + 1];
        value += neighValue;

        neighValue = image.ptr(x)[y + 1];
        value += (neighValue + neighValue);

        neighValue = image.ptr(x + 1)[y + 1];
        value += neighValue;

        switch (ddepth)
        {
        case CV_8U:
            res.at<uchar>(x - 1, y - 1) = cv::saturate_cast<uchar>(value);
            break;
        case CV_16S:
            res.at<int16_t>(x - 1, y - 1) = cv::saturate_cast<int16_t>(value);
            break;
        default:
            res.at<float>(x - 1, y - 1) = value;
            break;
        }
    }

    cv::Mat applySobelToChannel(const cv::Mat &channel, uint ddepth)
    {
        uint r = channel.rows, c = channel.cols;
        cv::Mat result = cv::Mat::zeros(r - 2, c - 2, ddepth);

        auto range = cv::Range(0, r * c);
        cv::parallel_for_(range, [&channel, &result, c, ddepth](const cv::Range &rng)
                          {
                            for (int k = rng.start; k != rng.end; ++k)
                            {
                                int x = (k / c) + 1;
                                int y = (k % c) + 1;
                                sobelKernel(channel, result, x, y, ddepth);
                            } });

        return result;
    }

    cv::Mat applySobel(const cv::Mat &image, uint ddepth = CV_8U,
                       cv::BorderTypes borderType = cv::BorderTypes::BORDER_REFLECT)
    {
        cv::Mat paddedImg;
        cv::copyMakeBorder(image, paddedImg, 1, 1, 1, 1, borderType);

        std::vector<cv::Mat> imgChannels, resChannels;
        cv::split(paddedImg, imgChannels);

        std::transform(imgChannels.begin(), imgChannels.end(), std::back_inserter(resChannels),
                       [ddepth](const cv::Mat &channel)
                       {
                           return applySobelToChannel(channel, ddepth);
                       });

        cv::Mat result;
        cv::merge(resChannels, result);

        return result;
    }

    void sobelMain(const cv::Mat &image)
    {
        auto result = applySobel(image);

        imshow("Our Sobel Applied", result);
        imshow("Original Image", image);

        cv::Mat reference;
        cv::Sobel(image, reference, -1, 1, 0);
        imshow("Reference Sobel", reference);

        cv::waitKey();
    }
}