#include "connected_component.hpp"

#include <numeric>

namespace tud::cvlabs
{
    std::size_t ConnectedComponent::n_points() const noexcept
    {
        return pixels.size();
    }

    cv::Rect ConnectedComponent::bounding_box() const noexcept
    {
        auto [xmin, xmax] = std::minmax_element(pixels.begin(), pixels.end(),
                                                [](const cv::Point &pixel1, const cv::Point &pixel2)
                                                {
                                                    return pixel1.x < pixel2.x;
                                                });

        auto [ymin, ymax] = std::minmax_element(pixels.begin(), pixels.end(),
                                                [](const cv::Point &pixel1, const cv::Point &pixel2)
                                                {
                                                    return pixel1.y < pixel2.y;
                                                });

        return cv::Rect(xmin->x, ymin->y, xmax->x - xmin->x, ymax->y - ymin->y);
    }

    cv::Point2f ConnectedComponent::centroid() const noexcept
    {
        auto M00 = moment(0, 0);
        auto M10 = moment(1, 0);
        auto M01 = moment(0, 1);
        return cv::Point2f(M10 / M00, M01 / M00);
    }

    float ConnectedComponent::moment(uint p, uint q) const noexcept
    {
        return std::transform_reduce(pixels.begin(), pixels.end(), 0, std::plus<>{},
                                     [p, q](const cv::Point &pixel)
                                     {
                                         int x = pixel.x;
                                         int y = pixel.y;
                                         return std::pow(x, p) * std::pow(y, q);
                                     });
    }

    float ConnectedComponent::eccentricity() const noexcept
    {
        // To Do
    }

    float ConnectedComponent::orientation() const noexcept
    {
        // To Do
    }

    void ConnectedComponent::add_point(int x, int y)
    {
        pixels.emplace_back(x, y);
    }

    void ConnectedComponent::add_point(const cv::Point &point)
    {
        pixels.push_back(point);
    }
}