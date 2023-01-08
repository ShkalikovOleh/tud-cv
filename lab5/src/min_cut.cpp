#include "min_cut.hpp"

#include <queue>
#include <map>
#include <algorithm>
#include <limits>

namespace tud::cvlabs
{
    void Graph::addEdge(Graph::idx_t from, Graph::idx_t to, float w)
    {
        vertices_[from].emplace_back(to, w);
    }

    Graph::EdgeList::const_iterator Graph::getEdge(Graph::idx_t from, Graph::idx_t to) const noexcept
    {
        auto &edges = vertices_[from];
        return std::find_if(edges.begin(), edges.end(),
                            [to](const OutEdge &edge)
                            {
                                return edge.adj_idx == to;
                            });
    }

    Graph::EdgeList::iterator Graph::getEdge(Graph::idx_t from, Graph::idx_t to) noexcept
    {
        auto &edges = vertices_[from];
        return std::find_if(edges.begin(), edges.end(),
                            [to](const OutEdge &edge)
                            {
                                return edge.adj_idx == to;
                            });
    }

    Graph::idx_t Graph::addVertex()
    {
        vertices_.push_back(EdgeList{});
        return size() - 1;
    }

    using BFSCallback = std::function<bool(const Graph &, Graph::idx_t, const std::map<Graph::idx_t, Graph::idx_t> &)>;

    void BFS(const Graph &resGraph, const Graph::idx_t s, BFSCallback f)
    {
        std::map<Graph::idx_t, Graph::idx_t> parents;
        std::queue<Graph::idx_t> queue;

        queue.push(s);
        parents.emplace(s, s);

        while (!queue.empty())
        {
            auto v = queue.front();
            queue.pop();

            if (f(resGraph, v, parents))
                break;

            for (auto &&edge : resGraph[v])
            {
                if (edge.w > 0 && parents.find(edge.adj_idx) == parents.end())
                {
                    queue.push(edge.adj_idx);
                    parents.emplace(edge.adj_idx, v);
                }
            }
        }
    }

    // BFS from s until we reach t
    bool find_aug_path(const Graph &resGraph, const Graph::idx_t s,
                       const Graph::idx_t t, std::vector<Graph::idx_t> &outRevPath, float &minCapacity)
    {
        outRevPath.clear();
        minCapacity = std::numeric_limits<float>::max();

        BFS(resGraph, s,
            [t, s, &outRevPath, &minCapacity](const Graph &resGraph, Graph::idx_t v, const std::map<Graph::idx_t, Graph::idx_t> &parents)
            {
                if (v == t)
                {
                    while (v != s)
                    {
                        outRevPath.push_back(v);
                        auto temp = parents.find(v)->second;

                        float w = resGraph.getEdge(temp, v)->w;
                        if (w < minCapacity)
                            minCapacity = w;

                        v = temp;
                    }
                    outRevPath.push_back(s);
                    return true;
                }
                return false;
            });

        if (!outRevPath.empty())
            return true;
        else
            return false;
    }

    std::vector<Graph::idx_t> min_cut(Graph &resGraph, const Graph::idx_t s, const Graph::idx_t t)
    {
        std::vector<Graph::idx_t> reversePath;
        float minCapacity;

        while (find_aug_path(resGraph, s, t, reversePath, minCapacity))
        {
            for (int i = 0; i < reversePath.size() - 1; ++i)
            {
                auto reverse = resGraph.getEdge(reversePath[i], reversePath[i + 1]);
                auto forward = resGraph.getEdge(reversePath[i + 1], reversePath[i]);
                forward->w -= minCapacity;
                reverse->w += minCapacity;
            }
        }

        // min-cut is an all vertices which we can reach from s in res graph
        std::vector<Graph::idx_t> S;
        BFS(resGraph, s,
            [&S](const Graph &resGraph, Graph::idx_t v, const std::map<Graph::idx_t, Graph::idx_t> &parents)
            {
                S.push_back(v);
                return false;
            });

        return S;
    }

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

        auto zeroClassIdxs = min_cut(graph, s, t);
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
