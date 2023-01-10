#include "graph.hpp"

#include <algorithm>

namespace tud::cvlabs
{
    void Graph::addEdge(Graph::idx_t from, Graph::idx_t to, float w)
    {
        edges_[from].emplace_back(to, w);
    }

    Graph::EdgeList::const_iterator Graph::getEdge(Graph::idx_t from, Graph::idx_t to) const noexcept
    {
        auto &edges = edges_[from];
        return std::find_if(edges.begin(), edges.end(),
                            [to](const OutEdge &edge)
                            {
                                return edge.adj_idx == to;
                            });
    }

    Graph::EdgeList::iterator Graph::getEdge(Graph::idx_t from, Graph::idx_t to) noexcept
    {
        auto &edges = edges_[from];
        return std::find_if(edges.begin(), edges.end(),
                            [to](const OutEdge &edge)
                            {
                                return edge.adj_idx == to;
                            });
    }

    Graph::idx_t Graph::addVertex()
    {
        edges_.push_back(EdgeList{});
        return size() - 1;
    }
}