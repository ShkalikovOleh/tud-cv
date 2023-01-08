#include <opencv2/opencv.hpp>

#include <vector>
#include <functional>

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

            OutEdge(idx_t idx, float w) : adj_idx(idx), w(w) {};
        };

        using EdgeList = std::vector<OutEdge>;

    public:
        Graph(std::size_t size) : vertices_(size) {}

        inline std::size_t size() const noexcept
        {
            return vertices_.size();
        }

        inline const EdgeList &operator[](idx_t vIdx) const noexcept
        {
            return vertices_[vIdx];
        }

        idx_t addVertex();
        void addEdge(idx_t from, idx_t to, float w);

        EdgeList::const_iterator getEdge(idx_t from, idx_t to) const noexcept;
        EdgeList::iterator getEdge(idx_t from, idx_t to) noexcept;

    private:
        std::vector<EdgeList> vertices_;
    };

    std::vector<Graph::idx_t> min_cut(Graph &resGraph, const Graph::idx_t s, const Graph::idx_t t);

    using cfunc = std::function<float(int, int, cv::Vec3b)>;
    using ccfunc = std::function<float(int, int, cv::Vec3b, int, int, cv::Vec3b)>;

    cv::Mat min_cut_classify(const cv::Mat &image, const cfunc &c, const ccfunc &cc);
}
