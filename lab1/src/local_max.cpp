#include <opencv2/imgproc.hpp>
#include "local_max.hpp"

namespace tud::cvlabs
{
    inline void naiveLocalMaxRawPtrKernel(const cv::Mat &image, const cv::Mat &neigh_mask,
                                          int rstart, int rend, cv::Mat &result)
    {
        int ncol = image.cols, nrow = image.rows;

        double maxValue;
        for (int i = rstart; i < rend; ++i)
        {
            for (int j = 1; j < ncol - 1; ++j)
            {
                auto currValue = image.ptr(i)[j];
                uchar isLocalMax = 255;

                for (int k = -1; k <= 1; ++k)
                {
                    auto imagePtr = image.ptr(i + k);
                    auto neighPtr = neigh_mask.ptr(1 + k);

                    for (int l = -1; l <= 1; ++l)
                    {
                        if (imagePtr[j + l] * neighPtr[1 + l] >= currValue)
                        {
                            isLocalMax = 0;
                            break;
                        }
                    }

                    if (!isLocalMax)
                        break;
                }

                result.ptr(i)[j] = isLocalMax;
            }
        }
    }

    cv::Mat localMaxNaive(const cv::Mat &image, const cv::Mat &neigh_mask)
    {
        int c = image.cols, r = image.rows;
        cv::Mat result(r, c, CV_8UC1);
        naiveLocalMaxRawPtrKernel(image, neigh_mask, 1, r - 1, result);
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
                naiveLocalMaxRawPtrKernel(image, neigh_mask, range.start, range.end, result);
            });

        return result;
    }

    cv::Mat localMaxDilate(const cv::Mat &image, const cv::Mat &neigh_mask)
    {
        cv::Mat result;

        /*
            Dilation replaces anchor (center) pixel with maximum value according to the mask
            In our mask, center (current pixel) is 0 so we will get maximum of neighbours
            More info: https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html
        */
        cv::dilate(image, result, neigh_mask);
        result = image > result; // local max must be greater than any neighbours

        return result;
    }
}