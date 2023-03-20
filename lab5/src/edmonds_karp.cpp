#include "min_cut.hpp"

#include <queue>
#include <map>
#include <limits>

namespace tud::cvlabs
{
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

    std::vector<Graph::idx_t> min_cut_edmonds_karp(Graph &resGraph, const Graph::idx_t s, const Graph::idx_t t)
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
}