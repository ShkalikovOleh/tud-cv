#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    class ConnectedComponent
    {
    public:
        cv::Rect bounding_box() const noexcept;
        cv::Point2f centroid() const noexcept;
        float moment(uint p, uint q) const noexcept;
        float eccentricity() const noexcept;
        float orientation() const noexcept;
        std::size_t n_points() const noexcept;

        void add_point(int x, int y);
        void add_point(const cv::Point &point);

    private:
        std::vector<cv::Point> pixels;
    };

    std::vector<ConnectedComponent> floodFilling(const cv::Mat &image);

    std::vector<ConnectedComponent> twoPass(const cv::Mat &image);
}