#include "pixel_classification.hpp"

#include <queue>
#include <unordered_set>
#include <numeric>

namespace tud::cvlabs
{
    uchar getBestClass(cv::Point point, const cv::Mat &mask, const cv::Mat &image, const cfunc &c, const ccfunc &cc)
    {
        int x = point.x, y = point.y, nrows = image.rows, ncols = image.cols;

        int xs = x == 0 ? x : x - 1;
        int ys = y == 0 ? y : y - 1;
        int h = y == nrows - 1 ? 2 : 3;
        int w = x == ncols - 1 ? 2 : 3;
        cv::Mat neighs = image(cv::Rect(xs, ys, w, h));

        auto v = image.at<cv::Vec3b>(point);
        auto phi = c(x, y, v);

        for (auto it = neighs.begin<cv::Vec3b>(); it != neighs.end<cv::Vec3b>(); ++it)
        {
            auto pos = it.pos();
            if (pos != point)
            {
                auto ccVal = cc(x, y, v, pos.x, pos.y, image.at<cv::Vec3b>(pos));
                auto yw = mask.at<uchar>(pos); // neighbour's label

                std::transform(ccVal.begin(), ccVal.end(), phi.begin(), phi.begin(), std::plus<>{}); // phi += cc_vw
                phi[yw] -= ccVal[yw];                                                                // we don't penaltize if we have the same class label
            }
        }

        auto minIt = std::min_element(phi.begin(), phi.end());

        return std::distance(phi.begin(), minIt);
    }

    cv::Mat classify(const cv::Mat &image, const cfunc &c, const ccfunc &cc)
    {
        int nrows = image.rows, ncols = image.cols;

        cv::Mat result(nrows, ncols, CV_8UC1);
        std::queue<cv::Point> W;
        std::unordered_set<int> enqueued;

        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                W.emplace(j, i);
                enqueued.insert(i * ncols + j);
            }
        }

        while (!W.empty())
        {
            auto v = W.front();
            W.pop();
            auto y = v.x, x = v.y;

            enqueued.erase(x * ncols + y);

            // auto &currClass = result.ptr<uchar>(x)[y];
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

                for (int k = -ks; k <= ke; ++k)
                {
                    for (int l = -ls; l <= le; ++l)
                    {
                        if (k == 0 && l == 0)
                            break;

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
