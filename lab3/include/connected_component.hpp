#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    class ConnectedComponent
    {
    public:
        cv::Rect bounding_box() const noexcept;
        cv::Point centroid() const noexcept;

        long moment(uint p, uint q) const noexcept;
        template <typename T>
        auto moment(uint p, uint q, const cv::Mat &image) const;

        long central_moment(uint p, uint q) const noexcept;
        template <typename T>
        auto central_moment(uint p, uint q, const cv::Mat &image) const;

        float eccentricity() const noexcept;
        template <typename T>
        float eccentricity(const cv::Mat &image) const;

        float orientation() const noexcept;
        template <typename T>
        float orientation(const cv::Mat &image) const;

        std::size_t n_points() const noexcept;

        void add_point(int x, int y);
        void add_point(const cv::Point &point);

        auto begin() { return pixels.begin(); }
        auto end() { return pixels.end(); }

    private:
        std::vector<cv::Point> pixels;
    };

    std::tuple<cv::Mat, std::vector<ConnectedComponent>> floodFilling(const cv::Mat &image);

    std::tuple<cv::Mat, std::vector<ConnectedComponent>> twoPass(const cv::Mat &image);
}