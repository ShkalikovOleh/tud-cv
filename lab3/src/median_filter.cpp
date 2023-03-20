#include <algorithm>

#include "median_filter.hpp"

namespace tud::cvlabs
{
    cv::Mat medianFilter(const cv::Mat &image, int kw, int kh, cv::BorderTypes borderType)
    {
        cv::Mat paddedImg;
        int w = kw / 2, h = kh / 2;
        cv::copyMakeBorder(image, paddedImg, h, h, w, w, borderType);

        cv::Mat result(image.rows, image.cols, CV_8UC1);

        cv::parallel_for_(cv::Range(h, image.rows + h),
                          [&](cv::Range rng)
                          {
                              int cols = image.cols;
                              for (int r = rng.start; r < rng.end; ++r)
                              {
                                  for (int c = w; c < cols + w; ++c)
                                  {
                                      auto subImage = paddedImg(cv::Range(r - h, r + h), cv::Range(c - w, c + w)); // just view, not copy
                                      auto medIt = subImage.begin<uchar>() + (kw * kh) / 2 + 1;
                                      std::nth_element(subImage.begin<uchar>(), medIt, subImage.end<uchar>());
                                      result.ptr<uchar>(r - h)[c - w] = *medIt;
                                  }
                              }
                          });

        return result;
    }
}