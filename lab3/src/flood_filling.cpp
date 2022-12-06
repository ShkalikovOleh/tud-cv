#include "connected_component.hpp"

#include <stack>
#include <unordered_set>

namespace tud::cvlabs
{
    std::vector<ConnectedComponent> floodFilling(const cv::Mat &image)
    {
        std::stack<cv::Point> stack;
        std::unordered_set<int> visited;
        visited.reserve(image.rows * image.cols);

        std::vector<ConnectedComponent> result;

        for (int r = 0; r < image.rows; ++r)
        {
            for (int c = 0; c < image.cols; ++c)
            {
                if (image.ptr<uchar>(r)[c] && visited.find(r * image.cols + c) == visited.end())
                {
                    ConnectedComponent component;

                    stack.emplace(cv::Point(c, r));
                    visited.insert(r * image.cols + c);
                    while (!stack.empty())
                    {
                        auto point = stack.top();
                        stack.pop();

                        if (image.at<uchar>(point))
                        {
                            component.add_point(point);

                            if (point.y > 0 &&
                                visited.find((point.y - 1) * image.cols + point.x) == visited.end())
                            {
                                stack.emplace(point.x, point.y - 1); // top
                                visited.insert((point.y - 1) * image.cols + point.x);
                            }
                            if (point.x > 0 && point.y > 0 &&
                                visited.find((point.y - 1) * image.cols + point.x - 1) == visited.end())
                            {
                                stack.emplace(point.x - 1, point.y - 1); // left top
                                visited.insert((point.y - 1) * image.cols + point.x - 1);
                            }
                            if (point.y > 0 && point.x < image.rows - 1 &&
                                visited.find((point.y - 1) * image.cols + point.x + 1) == visited.end())
                            {
                                stack.emplace(point.x + 1, point.y - 1); // right top
                                visited.insert((point.y - 1) * image.cols + point.x + 1);
                            }
                            if (point.x > 0 &&
                                visited.find(point.y * image.cols + point.x - 1) == visited.end())
                            {
                                stack.emplace(point.x - 1, point.y); // left
                                visited.insert(point.y * image.cols + point.x - 1);
                            }
                            if (point.x < image.rows - 1 &&
                                visited.find(point.y * image.cols + point.x + 1) == visited.end())
                            {
                                stack.emplace(point.x + 1, point.y); // right
                                visited.insert(point.y * image.cols + point.x + 1);
                            }
                            if (point.y < image.cols - 1 &&
                                visited.find((point.y + 1) * image.cols + point.x) == visited.end())
                            {
                                stack.emplace(point.x, point.y + 1); // bottom
                                visited.insert((point.y + 1) * image.cols + point.x);
                            }
                            if (point.x > 0 && point.y < image.cols - 1 &&
                                visited.find((point.y + 1) * image.cols + point.x - 1) == visited.end())
                            {
                                stack.emplace(point.x - 1, point.y + 1); // left bottom
                                visited.insert((point.y + 1) * image.cols + point.x - 1);
                            }
                            if (point.y < image.cols - 1 && point.x < image.rows - 1 &&
                                visited.find((point.y + 1) * image.cols + point.x + 1) == visited.end())
                            {
                                stack.emplace(point.x + 1, point.y + 1); // right bottom
                                visited.insert((point.y + 1) * image.cols + point.x + 1);
                            }
                        }
                    }

                    result.push_back(component);
                }
            }
        }
        return result;
    }
}