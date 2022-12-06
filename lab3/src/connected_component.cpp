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

        return cv::Rect(xmin->x, ymin->y, xmax->x - xmin->x + 1, ymax->y - ymin->y + 1);
    }

    cv::Point ConnectedComponent::centroid() const noexcept
    {
        auto M00 = moment(0, 0);
        auto M10 = moment(1, 0);
        auto M01 = moment(0, 1);
        return cv::Point(M10 / M00, M01 / M00);
    }

    long ConnectedComponent::moment(uint p, uint q) const noexcept
    {
        return std::transform_reduce(pixels.begin(), pixels.end(), 0l, std::plus<>{},
                                     [p, q](const cv::Point &pixel)
                                     {
                                         int x = pixel.x;
                                         int y = pixel.y;
                                         return std::pow(x, p) * std::pow(y, q);
                                     });
    }

    template <typename T>
    auto ConnectedComponent::moment(uint p, uint q, const cv::Mat &image) const
    {
        using ret_type = std::common_type_t<T, long>;
        return std::transform_reduce(pixels.begin(), pixels.end(), static_cast<ret_type>(0), std::plus<>{},
                                     [&image, p, q](const cv::Point &pixel)
                                     {
                                         int x = pixel.x;
                                         int y = pixel.y;
                                         return image.ptr<T>(x)[y] * std::pow(x, p) * std::pow(y, q);
                                     });
    }

    long ConnectedComponent::central_moment(uint p, uint q) const noexcept
    {
        auto c = centroid();
        return std::transform_reduce(pixels.begin(), pixels.end(), 0l, std::plus<>{},
                                     [c, p, q](const cv::Point &pixel)
                                     {
                                         int x = pixel.x;
                                         int y = pixel.y;
                                         return std::pow(x - c.x, p) * std::pow(y - c.y, q);
                                     });
    }

    template <typename T>
    auto ConnectedComponent::central_moment(uint p, uint q, const cv::Mat &image) const
    {
        using ret_type = std::common_type_t<T, long>;
        auto c = centroid();
        return std::transform_reduce(pixels.begin(), pixels.end(), static_cast<ret_type>(0), std::plus<>{},
                                     [&image, c, p, q](const cv::Point &pixel)
                                     {
                                         int x = pixel.x;
                                         int y = pixel.y;
                                         return image.ptr<T>(x)[y] * std::pow(x - c.x, p) * std::pow(y - c.y, q);
                                     });
    }

    float ConnectedComponent::eccentricity() const noexcept
    {
        auto u11 = central_moment(1, 1);
        auto u02 = central_moment(0, 2);
        auto u20 = central_moment(2, 0);
        return (std::pow(u20 - u02, 2) + 4 * std::pow(u11, 2)) / std::pow(u20 + u02, 2);
    }

    template <typename T>
    float ConnectedComponent::eccentricity(const cv::Mat &image) const
    {
        auto u11 = central_moment<T>(1, 1, image);
        auto u02 = central_moment<T>(0, 2, image);
        auto u20 = central_moment<T>(2, 0, image);
        return (std::pow(u20 - u02, 2) + 4 * std::pow(u11, 2)) / std::pow(u20 + u02, 2);
    }

    float ConnectedComponent::orientation() const noexcept
    {
        auto u11 = static_cast<float>(central_moment(1, 1));
        auto u02 = central_moment(0, 2);
        auto u20 = central_moment(2, 0);
        return 0.5 * std::atan(u11 / (u20 - u02));
    }

    template <typename T>
    float ConnectedComponent::orientation(const cv::Mat &image) const
    {
        auto u11 = static_cast<float>(central_moment<T>(1, 1, image));
        auto u02 = central_moment<T>(0, 2, image);
        auto u20 = central_moment<T>(2, 0, image);
        return 0.5 * std::atan(u11 / (u20 - u02));
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