#include <opencv2/imgproc.hpp>
#include "local_max.hpp"

namespace tud::cvlabs
{
    inline void naiveLocalMaxKernel(const cv::Mat &image, const cv::Mat &neigh_mask,
                                    int rstart, int rend, cv::Mat &result)
    {
        int ncol = image.cols, nrow = image.rows;

        double maxValue;
        for (int i = rstart; i < rend; ++i)
        {
            auto rowPtr = image.ptr(i);

            for (int j = 1; j < ncol - 1; ++j)
            {
                auto slice = image(cv::Range(i - 1, i + 2), cv::Range(j - 1, j + 2)); // get 3x3 region around this pixel

                cv::minMaxLoc(slice, nullptr, &maxValue, nullptr, nullptr, neigh_mask); // get max value of neighbours

                if (maxValue < rowPtr[j]) // check whether it is a local max
                {
                    result.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    cv::Mat localMaxNaive(const cv::Mat &image, const cv::Mat &neigh_mask)
    {
        int c = image.cols, r = image.rows;
        cv::Mat result(r, c, CV_8UC1);
        naiveLocalMaxKernel(image, neigh_mask, 1, r - 1, result);
        return result;
    }

    cv::Mat localMaxNaiveParallel(const cv::Mat &image, const cv::Mat &neigh_mask)
    {
        int c = image.cols, r = image.rows;
        cv::Mat result(r, c, CV_8UC1);

        // parallelize over the rows
        cv::parallel_for_(
            cv::Range(1, r - 1),
            [&](const cv::Range &range)
            {
                naiveLocalMaxKernel(image, neigh_mask, range.start, range.end, result);
            });

        return result;
    }

    cv::Mat localMaxDilate(const cv::Mat &image, const cv::Mat &neigh_mask)
    {
        cv::Mat result;

        /*
            Dilation will replace anchor (center) pixel with maximum value according to the mask
            In our mask center (current pixel is 0) so we will get maximum of neighbours
            More info: https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html
        */
        cv::dilate(image, result, neigh_mask);
        result = image > result; // local max must be greater than any neighbours

        return result;
    }
}