#include <algorithm>
#include <vector>

#include "sobel_filter.hpp"

namespace tud::cvlabs
{
    cv::Mat applySobel(const cv::Mat &image, uint ddepth)
    {
        cv::Mat floatImg;
        image.convertTo(floatImg, CV_32F); // float helps to avoid overflow

        int r = floatImg.rows, c = floatImg.cols, ch = floatImg.channels();

        // applying of derivative (-1 0 1) along x axis
        cv::Mat derX(r, c, CV_32FC(ch));
        derX.col(0) = floatImg.col(1);
        derX.col(c - 1) = -floatImg.col(c - 2);
        cv::parallel_for_(cv::Range(1, c - 1), [&floatImg, &derX](const cv::Range &rng)
                          {
                            for (int k = rng.start; k != rng.end; ++k)
                            {
                                derX.col(k) = floatImg.col(k + 1) - floatImg.col(k - 1);
                            } });

        // applying of (1 1 0) over y axis
        cv::Mat smoothY1(r + 1, c, CV_32FC(ch));
        smoothY1.row(0) = derX.row(0);
        smoothY1.row(r) = derX.row(r - 1);
        cv::parallel_for_(cv::Range(1, r), [&derX, &smoothY1](const cv::Range &rng)
                          {
                            for (int k = rng.start; k != rng.end; ++k)
                            {
                                smoothY1.row(k) = derX.row(k) + derX.row(k - 1);
                            } });

        // applying of (0 1 1) over y axis
        cv::Mat result(r, c, CV_32FC(ch));
        cv::parallel_for_(cv::Range(0, r), [&smoothY1, &result, ddepth](const cv::Range &rng)
                          {
                            for (int k = rng.start; k != rng.end; ++k)
                            {
                                result.row(k) = smoothY1.row(k) + smoothY1.row(k + 1);
                            } });

        result.convertTo(result, ddepth);

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