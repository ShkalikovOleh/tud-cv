#include "min_cut.hpp"

namespace tud::cvlabs
{
    cv::Mat min_cut_classify(const cv::Mat &image, const cfunc &c, const ccfunc &cc)
    {
        int nrows = image.rows, ncols = image.cols;

        Graph graph{static_cast<Graph::idx_t>(nrows * ncols) + 2};

        Graph::idx_t s = nrows * ncols;
        Graph::idx_t t = nrows * ncols + 1;

        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                Graph::idx_t idx = i * ncols + j;

                auto &pixel = image.at<cv::Vec3b>(i, j);
                auto cval = c(i, j, pixel);
                if (cval > 0)
                {
                    graph.addEdge(s, idx, cval);
                    graph.addEdge(idx, s, 0);
                }
                else
                {
                    graph.addEdge(idx, t, -cval);
                    graph.addEdge(t, idx, 0);
                }

                int ks = i > 0 ? -1 : 0;
                int ke = i < nrows - 1 ? 1 : 0;
                int ls = j > 0 ? -1 : 0;
                int le = j < ncols - 1 ? 1 : 0;
                for (int k = ks; k <= ke; ++k)
                {
                    for (int l = ls; l <= le; ++l)
                    {
                        if (k != 0 || l != 0)
                        {
                            Graph::idx_t neigh_idx = (i + k) * ncols + (j + l);
                            auto ccval = cc(i, j, pixel, i + k, j + l, image.at<cv::Vec3b>(i + k, j + l));
                            graph.addEdge(idx, neigh_idx, ccval);
                        }
                    }
                }
            }
        }

        auto zeroClassIdxs = min_cut_boykov_kolmogorov(graph, s, t);
        zeroClassIdxs.erase(std::remove(zeroClassIdxs.begin(), zeroClassIdxs.end(), s));
        cv::Mat result{nrows, ncols, CV_8UC1, cv::Scalar{1}};

        for (auto &idx : zeroClassIdxs)
        {
            auto i = idx / ncols;
            auto j = idx % ncols;

            result.at<uchar>(i, j) = 0;
        }

        return result;
    }
}
