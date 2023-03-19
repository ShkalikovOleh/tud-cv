#pragma once
#include <vector>
#include <numeric>

#include <opencv2/opencv.hpp>

namespace tud::cvlabs
{
    class ConnectedComponent
    {
    public:
        cv::Rect bounding_box() const noexcept
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

        long moment(uint p, uint q) const noexcept
        {
            return std::transform_reduce(pixels.begin(), pixels.end(), 0l, std::plus<>{},
                                         [p, q](const cv::Point &pixel)
                                         {
                                             int x = pixel.y;
                                             int y = pixel.x;
                                             return std::pow(x, p) * std::pow(y, q);
                                         });
        }

        template <typename T>
        std::common_type_t<T, long> moment(uint p, uint q, const cv::Mat &image) const
        {
            using ret_type = std::common_type_t<T, long>;
            return std::transform_reduce(pixels.begin(), pixels.end(), static_cast<ret_type>(0), std::plus<>{},
                                         [&image, p, q](const cv::Point &pixel)
                                         {
                                             int x = pixel.y;
                                             int y = pixel.x;
                                             auto t = image.ptr<T>(x)[y];
                                             return image.ptr<T>(x)[y] * std::pow(x, p) * std::pow(y, q);
                                         });
        }

        long central_moment(uint p, uint q) const noexcept
        {
            auto c = centroid();
            return std::transform_reduce(pixels.begin(), pixels.end(), 0l, std::plus<>{},
                                         [c, p, q](const cv::Point &pixel)
                                         {
                                             int x = pixel.y;
                                             int y = pixel.x;
                                             return std::pow(x - c.x, p) * std::pow(y - c.y, q);
                                         });
        }

        template <typename T>
        std::common_type_t<T, long> central_moment(uint p, uint q, const cv::Mat &image) const
        {
            using ret_type = std::common_type_t<T, long>;
            auto c = centroid<T>(image);
            return std::transform_reduce(pixels.begin(), pixels.end(), static_cast<ret_type>(0), std::plus<>{},
                                         [&image, c, p, q](const cv::Point &pixel)
                                         {
                                             int x = pixel.y;
                                             int y = pixel.x;
                                             return image.ptr<T>(x)[y] * std::pow(x - c.x, p) * std::pow(y - c.y, q);
                                         });
        }

        cv::Point centroid() const noexcept
        {
            auto M00 = moment(0, 0);
            auto M10 = moment(1, 0);
            auto M01 = moment(0, 1);
            return cv::Point(M10 / M00, M01 / M00);
        }

        template <typename T>
        cv::Point centroid(const cv::Mat &image) const
        {
            auto M00 = moment<T>(0, 0, image);
            auto M10 = moment<T>(1, 0, image);
            auto M01 = moment<T>(0, 1, image);
            return cv::Point(M10 / M00, M01 / M00);
        }

        float eccentricity() const noexcept
        {
            auto u11 = central_moment(1, 1);
            auto u02 = central_moment(0, 2);
            auto u20 = central_moment(2, 0);

            // consider that elementary components has eccentricity equal to 0
            if (pixels.size() < 2)
                return 0;

            return (std::pow(u20 - u02, 2) + 4 * std::pow(u11, 2)) / std::pow(u20 + u02, 2);
        }

        template <typename T>
        float eccentricity(const cv::Mat &image) const
        {
            auto u11 = central_moment<T>(1, 1, image);
            auto u02 = central_moment<T>(0, 2, image);
            auto u20 = central_moment<T>(2, 0, image);

            // consider that elementary components has eccentricity equal to 0
            if (pixels.size() < 2)
                return 0;

            return (std::pow(u20 - u02, 2) + 4 * std::pow(u11, 2)) / std::pow(u20 + u02, 2);
        }

        float orientation() const noexcept
        {
            auto u11 = static_cast<float>(central_moment(1, 1));
            auto u02 = central_moment(0, 2);
            auto u20 = central_moment(2, 0);
            return 0.5 * std::atan2(u11, u20 - u02);
        }

        template <typename T>
        float orientation(const cv::Mat &image) const
        {
            auto u11 = static_cast<float>(central_moment<T>(1, 1, image));
            auto u02 = central_moment<T>(0, 2, image);
            auto u20 = central_moment<T>(2, 0, image);
            return 0.5 * std::atan2(u11, u20 - u02);
        }

        std::size_t n_points() const noexcept
        {
            return pixels.size();
        }

        void add_point(int x, int y)
        {
            pixels.emplace_back(x, y);
        }

        void add_point(const cv::Point &point)
        {
            pixels.push_back(point);
        }

        std::vector<cv::Point>::iterator begin()
        {
            return pixels.begin();
        }

        std::vector<cv::Point>::iterator end()
        {
            return pixels.end();
        }

    private:
        std::vector<cv::Point> pixels;
    };

    std::tuple<cv::Mat, std::vector<ConnectedComponent>> floodFilling(const cv::Mat &image);
}