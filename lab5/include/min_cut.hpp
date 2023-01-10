#pragma once
#include <opencv2/opencv.hpp>

#include <functional>

#include "graph.hpp"

namespace tud::cvlabs
{
    std::vector<Graph::idx_t> min_cut_edmonds_karp(Graph &resGraph, const Graph::idx_t s, const Graph::idx_t t);

    using cfunc = std::function<float(int, int, cv::Vec3b)>;
    using ccfunc = std::function<float(int, int, cv::Vec3b, int, int, cv::Vec3b)>;
    cv::Mat min_cut_classify(const cv::Mat &image, const cfunc &c, const ccfunc &cc);
}
