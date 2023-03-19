#include "pixel_classification.hpp"

#include <queue>
#include <set>
#include <numeric>

namespace tud::cvlabs
{
    uchar getBestClass(cv::Point point, const cv::Mat &mask, const cv::Mat &image, const cfunc &c, const ccfunc &cc)
    {
        int x = point.y, y = point.x, nrows = image.rows, ncols = image.cols;

        auto v = image.at<cv::Vec3b>(point);
        auto phi = c(x, y, v);

        int ks = x > 0 ? -1 : 0;
        int ke = x < nrows - 1 ? 1 : 0;
        int ls = y > 0 ? -1 : 0;
        int le = y < ncols - 1 ? 1 : 0;

        for (int k = ks; k <= ke; ++k)
        {
            for (int l = ls; l <= le; ++l)
            {
                if (k == 0 && l == 0)
                    continue;

                auto w = *image.ptr<cv::Vec3b>(x + k, y + l);
                auto yw = *mask.ptr<uchar>(x + k, y + l);
                auto ccVal = cc(x, y, v, x + k, y + l, yw, w);

                // phi += cc_vw
                // std::transform(ccVal.begin(), ccVal.end(), phi.begin(), phi.begin(), std::plus<>{});
                for (int i = 0; i < ccVal.size(); ++i)
                    phi[i] += ccVal[i];
            }
        }

        auto minIt = std::min_element(phi.begin(), phi.end());

        return std::distance(phi.begin(), minIt); // return class idx
    }

    cv::Mat classify(const cv::Mat &image, const cfunc &c, const ccfunc &cc)
    {
        int nrows = image.rows, ncols = image.cols;

        cv::Mat result(nrows, ncols, CV_8UC1, cv::Scalar(0));
        std::queue<cv::Point> W;
        std::set<int> enqueued;

        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                W.emplace(j, i);
                enqueued.insert(i * ncols + j);

#ifdef BEST_POINTWISE_INIT
                auto v = image.at<cv::Vec3b>(i, j);
                auto phi = c(i, j, v);
                auto minIt = std::min_element(phi.begin(), phi.end());
                result.at<uchar>(i, j) = std::distance(phi.begin(), minIt);
#endif
            }
        }

        while (!W.empty())
        {
            auto v = W.front();
            W.pop();
            auto y = v.x, x = v.y;

            enqueued.erase(x * ncols + y);

            auto &currClass = result.at<uchar>(v);
            auto bestClass = getBestClass(v, result, image, c, cc);
            if (bestClass != currClass)
            {
                currClass = bestClass;

                // assume 8-neighborhood
                int ks = x > 0 ? -1 : 0;
                int ke = x < nrows - 1 ? 1 : 0;
                int ls = y > 0 ? -1 : 0;
                int le = y < ncols - 1 ? 1 : 0;

                for (int k = ks; k <= ke; ++k)
                {
                    for (int l = ls; l <= le; ++l)
                    {
                        if (k == 0 && l == 0)
                            continue;

                        auto idx = (x + k) * ncols + y + l;
                        if (enqueued.find(idx) == enqueued.end())
                        {
                            W.emplace(y + l, x + k);
                            enqueued.insert(idx);
                        }
                    }
                }
            }
        }

        return result;
    }
}
