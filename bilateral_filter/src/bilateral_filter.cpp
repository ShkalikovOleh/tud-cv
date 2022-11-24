#include <cmath>
#include <numeric>
#include <algorithm>
#include <type_traits>

#include "bilateral_filter.hpp"

namespace tud
{
    std::vector<float> get_distance_coeffs(int n, wfunc ds_func)
    {
        std::vector<float> ds((2 * n + 1) * (2 * n + 1));
        for (int i = -n, idx = 0; i <= n; ++i)
        {
            for (int j = -n; j <= n; ++j)
            {
                auto dist = std::sqrt(i * i + j * j); // std::hypot works slower on my machine
                if (dist <= n)
                {
                    ds[idx] = ds_func(dist);
                }
                ++idx;
            }
        }
        return ds;
    }

    std::vector<float> get_color_coeffs(wfunc dc_func, float scaleIdx, int bins)
    {
        std::vector<float> dc(bins);
        for (int i = 0; i < bins; ++i)
        {
            auto c = i / scaleIdx;
            dc[i] = dc_func(c);
        }
        return dc;
    }

    template <typename T>
    cv::Mat bilateral_filter_impl(const cv::Mat &paddedImg, const std::vector<float> &ds,
                                  const std::vector<float> &dc, int n, float dcIdxScale)
    {
        int r = paddedImg.rows - 2 * n, c = paddedImg.cols - 2 * n, ch = paddedImg.channels();

        cv::Mat result(r, c, paddedImg.type());

        cv::parallel_for_(
            cv::Range(0, r),
            [&](cv::Range rowRange)
            {
                using TRes = std::common_type_t<float, T>;
                using TDist = std::common_type_t<int, T>;

                std::vector<TRes> sum(ch);

                // x, y - is a range of coordinated of the result image
                for (int x = rowRange.start; x != rowRange.end; ++x)
                {
                    int X = x + n; // real current row index of the input image
                    const auto currRowPtr = paddedImg.ptr<T>(X);
                    for (int y = 0; y < c; ++y)
                    {
                        int Y = ch * (y + n); // real column index of the input image
                        const auto p0 = currRowPtr + Y;

                        TRes v = 0.f;                         // normalization factor
                        std::fill(sum.begin(), sum.end(), 0); // zeroize sum

                        for (int i = -n; i <= n; ++i)
                        {
                            const auto neighRowPtr = paddedImg.ptr<T>(X + i) + Y;
                            for (int j = -n; j <= n; ++j)
                            {
                                const auto p = neighRowPtr + ch * j;

                                // l1 norm computing (color distance)
                                TDist dist_c =
                                    std::transform_reduce(p, p + ch, p0,
                                                          0, std::plus<>{},
                                                          [](auto pk, auto p0k)
                                                          {
                                                              return std::abs(pk - p0k);
                                                          });

                                // weight calculation
                                float w = 0.;
                                if constexpr (std::is_same_v<T, uchar>)
                                {
                                    w = dc[dist_c] * ds[(i + n) * (2 * n + 1) + j + n];
                                }
                                else
                                {
                                    w = dc[dcIdxScale * dist_c] * ds[(i + n) * (2 * n + 1) + j + n];
                                }

                                // update norm and sum
                                v += w;
                                std::transform(p, p + ch, sum.begin(), sum.begin(),
                                               [w](auto pk, auto sumk)
                                               {
                                                   return sumk + w * pk;
                                               });
                            }
                        }

                        // write result
                        v = 1. / v;
                        auto resPtr = result.ptr<T>(x) + ch * y;
                        std::transform(sum.begin(), sum.end(), resPtr,
                                       [v](auto sumk)
                                       {
                                           return v * sumk;
                                       });
                    }
                }
            });

        return result;
    }

    cv::Mat bilateral_filter(const cv::Mat &image, uint n,
                             wfunc ds_func, wfunc dc_func)
    {
        cv::Mat paddedImg;
        cv::copyMakeBorder(image, paddedImg, n, n, n, n, cv::BorderTypes::BORDER_DEFAULT);

        auto ds = get_distance_coeffs(n, ds_func); // precomputed distance weights

        int nchannels = image.channels();
        int depth = image.depth();

        if (depth == CV_16F) // float16_t support is very limited in OpenCV (even arithmetic)
        {
            throw std::invalid_argument("Image type is not supported");
        }
        else if (depth == CV_8U)
        {
            auto dc = get_color_coeffs(dc_func, 1, 256 * nchannels);
            return bilateral_filter_impl<uchar>(paddedImg, ds, dc, n, 1);
        }
        else
        {
            double max, min;
            cv::minMaxLoc(image, &min, &max);

            const int bins = 2 << 10;
            float idxScale = bins / (max - min) / nchannels;

            auto dc = get_color_coeffs(dc_func, idxScale, bins);

            if (depth == CV_8S)
            {
                return bilateral_filter_impl<cv::int8_t>(paddedImg, ds, dc, n, idxScale);
            }
            else if (depth == CV_16S)
            {
                return bilateral_filter_impl<cv::int16_t>(paddedImg, ds, dc, n, idxScale);
            }
            else if (depth == CV_16U)
            {
                return bilateral_filter_impl<cv::uint16_t>(paddedImg, ds, dc, n, idxScale);
            }
            else if (depth == CV_32S)
            {
                return bilateral_filter_impl<cv::int32_t>(paddedImg, ds, dc, n, idxScale);
            }
            else if (depth == CV_32F)
            {
                return bilateral_filter_impl<float>(paddedImg, ds, dc, n, idxScale);
            }
            else // depth == CV_64F
            {
                return bilateral_filter_impl<double>(paddedImg, ds, dc, n, idxScale);
            }
        }
    }
}
