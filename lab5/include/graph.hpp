#pragma once
#include <vector>

namespace tud::cvlabs
{
    class Graph
    {
    public:
        using idx_t = std::size_t;

        struct OutEdge
        {
            idx_t adj_idx; // idx of the adjecent vertex
            float w;

            OutEdge(idx_t idx, float w) : adj_idx(idx), w(w){};
        };

        using EdgeList = std::vector<OutEdge>;

    public:
        Graph(std::size_t size) : edges_(size) {}

        inline std::size_t size() const noexcept
        {
            return edges_.size();
        }

        inline const EdgeList &operator[](idx_t vIdx) const noexcept
        {
            return edges_[vIdx];
        }

        idx_t addVertex();
        void addEdge(idx_t from, idx_t to, float w);

        EdgeList::const_iterator getEdge(idx_t from, idx_t to) const noexcept;
        EdgeList::iterator getEdge(idx_t from, idx_t to) noexcept;

    private:
        std::vector<EdgeList> edges_;
    };
}