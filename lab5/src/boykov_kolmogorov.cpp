#include "min_cut.hpp"

#include <deque>
#include <map>
#include <limits>

namespace tud::cvlabs
{
    enum class BKNodeType
    {
        S,
        T,
        FREE
    };

    inline float tree_cap(Graph::idx_t p, Graph::idx_t q,
                          const Graph &resGraph,
                          const std::vector<BKNodeType> &tree)
    {
        return tree[p] == BKNodeType::S ? resGraph.getEdge(p, q)->w : resGraph.getEdge(q, p)->w;
    }

    std::deque<Graph::idx_t> grow(const Graph &resGraph, std::deque<Graph::idx_t> &active,
                                  std::map<Graph::idx_t, Graph::idx_t> &parents,
                                  std::vector<BKNodeType> &tree,
                                  std::vector<BKNodeType> &origins,
                                  float &minCapacity)
    {
        minCapacity = std::numeric_limits<float>::max();
        std::deque<Graph::idx_t> outPath;

        while (!active.empty())
        {
            auto v = active.front();
            active.pop_front();

            for (auto &&edge : resGraph[v])
            {
                auto q = edge.adj_idx;
                float w = tree[v] == BKNodeType::S ? edge.w : resGraph.getEdge(q, v)->w;
                if (w > 0)
                {
                    auto qtype = tree[q];
                    if (qtype == BKNodeType::FREE)
                    {
                        tree[q] = tree[v];
                        parents.emplace(q, v);
                        origins[q] = origins[v];
                        active.push_back(q);
                    }
                    else if (qtype != tree[v])
                    {
                        auto s = qtype == BKNodeType::T ? v : q; // last S node
                        auto t = qtype == BKNodeType::T ? q : v; // last T node

                        minCapacity = resGraph.getEdge(s, t)->w;

                        while (true)
                        {
                            outPath.push_front(s);
                            auto pIt = parents.find(s);
                            if (pIt == parents.end())
                                break;

                            auto p = pIt->second;
                            float w = resGraph.getEdge(p, s)->w;
                            if (w < minCapacity)
                                minCapacity = w;

                            s = p;
                        }

                        while (true)
                        {
                            outPath.push_back(t);
                            auto pIt = parents.find(t);
                            if (pIt == parents.end())
                                break;

                            auto p = pIt->second;
                            float w = resGraph.getEdge(t, p)->w;
                            if (w < minCapacity)
                                minCapacity = w;

                            t = p;
                        }

                        return outPath;
                    }
                }
            }
        }

        return outPath;
    }

    std::vector<Graph::idx_t> augment(Graph &resGraph, const std::deque<Graph::idx_t> path,
                                      float flow, std::map<Graph::idx_t, Graph::idx_t> &parents,
                                      std::vector<BKNodeType> &tree,
                                      std::vector<BKNodeType> &origins)
    {
        std::vector<Graph::idx_t> orphans;

        for (int i = 0; i < path.size() - 1; ++i)
        {
            auto p = path[i];
            auto q = path[i + 1];

            auto forward = resGraph.getEdge(p, q);
            auto reverse = resGraph.getEdge(q, p);

            forward->w -= flow;
            reverse->w += flow;

            if (forward->w <= 0)
            {
                if (tree[p] == tree[q])
                {
                    auto v = tree[p] == BKNodeType::T ? p : q;
                    parents.erase(v);
                    orphans.push_back(v);

                    std::queue<Graph::idx_t> queue;
                    queue.push(v);
                    while (!queue.empty())
                    {
                        auto v = queue.front();
                        queue.pop();
                        origins[v] = BKNodeType::FREE;

                        for (auto &&edge : resGraph[v])
                        {
                            auto q = edge.adj_idx;
                            auto qIt = parents.find(q);
                            if (qIt != parents.end() && v == qIt->second)
                            {
                                queue.push(q);
                            }
                        }
                    }
                }
            }
        }

        return orphans;
    }

    void adopt(const Graph &resGraph, std::vector<Graph::idx_t> &orphans,
               std::map<Graph::idx_t, Graph::idx_t> &parents,
               std::vector<BKNodeType> &tree,
               std::vector<BKNodeType> &origins,
               std::deque<Graph::idx_t> &active)
    {
        while (!orphans.empty())
        {
            auto p = orphans.back();
            orphans.pop_back();

            auto adjList = resGraph[p];
            auto validParent = std::find_if(adjList.begin(), adjList.end(),
                                            [&resGraph, &tree, &parents, &origins, p](const Graph::OutEdge &edge)
                                            {
                                                auto q = edge.adj_idx;
                                                // auto qpCap = resGraph.getEdge(q, p)->w;
                                                auto qpCap = tree_cap(q, p, resGraph, tree);
                                                return tree[p] == tree[q] && qpCap > 0 && origins[q] != BKNodeType::FREE;
                                            });

            if (validParent != adjList.end())
            {
                parents.emplace(p, validParent->adj_idx);
            }
            else
            {
                for (auto &&edge : adjList)
                {
                    auto q = edge.adj_idx;
                    if (tree[q] == tree[p])
                    {
                        auto qp = resGraph.getEdge(q, p);

                        if (tree_cap(q, p, resGraph, tree) > 0)
                        // if (qp->w > 0)
                        {
                            auto qIt = std::find(active.begin(), active.end(), q);
                            if (qIt == active.end())
                                active.push_back(q);
                        }

                        auto qParIt = parents.find(q);
                        if (qParIt != parents.end() && qParIt->second == p)
                        {
                            orphans.push_back(q);
                            parents.erase(q);
                        }
                    }
                }

                tree[p] = BKNodeType::FREE;

                // remove-erase idiom
                auto endIt = std::remove(active.begin(), active.end(), p);
                active.erase(endIt, active.end());
            }
        }
    }

    std::vector<Graph::idx_t> min_cut_boykov_kolmogorov(Graph &resGraph, const Graph::idx_t s, const Graph::idx_t t)
    {
        std::deque<Graph::idx_t> active;
        std::map<Graph::idx_t, Graph::idx_t> parents;
        std::vector<BKNodeType> tree(resGraph.size(), BKNodeType::FREE);
        std::vector<BKNodeType> origins(resGraph.size(), BKNodeType::FREE);
        float minCapacity;

        tree[s] = BKNodeType::S;
        tree[t] = BKNodeType::T;
        origins[s] = BKNodeType::S;
        origins[t] = BKNodeType::T;
        active.push_back(s);
        active.push_back(t);

        while (true)
        {
            auto path = grow(resGraph, active, parents, tree, origins, minCapacity);
            if (path.empty())
                break;
            auto orphans = augment(resGraph, path, minCapacity, parents, tree, origins);
            adopt(resGraph, orphans, parents, tree, origins, active);
        }

        std::vector<Graph::idx_t> S;
        for (std::size_t i = 0; i < tree.size(); ++i)
        {
            if (tree[i] == BKNodeType::S)
                S.push_back(i);
        }

        return S;
    }
}