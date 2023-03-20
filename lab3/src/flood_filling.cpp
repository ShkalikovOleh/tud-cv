#include "connected_component.hpp"

#include <stack>
#include <unordered_set>

namespace tud::cvlabs
{
    std::tuple<cv::Mat, std::vector<ConnectedComponent>> floodFilling(const cv::Mat &image)
    {
        std::stack<cv::Point> stack;

        std::vector<ConnectedComponent> result;
        cv::Mat labels(image.rows, image.cols, CV_32SC1, cv::Scalar(0));

        int n = 1;

        for (int r = 0; r < image.rows; ++r)
        {
            for (int c = 0; c < image.cols; ++c)
            {
                if (image.ptr<uchar>(r)[c] && labels.ptr<int>(r)[c] == 0)
                {
                    ConnectedComponent component;

                    stack.emplace(c, r);
                    while (!stack.empty()) // DFS with avoiding double queueing
                    {
                        auto point = stack.top();
                        stack.pop();

                        if (image.at<uchar>(point))
                        {
                            component.add_point(point);
                            labels.at<int>(point) = n;

                            if (point.y > 0 && labels.ptr<int>(point.y - 1)[point.x] == 0)
                            {
                                stack.emplace(point.x, point.y - 1); // top
                                labels.ptr<int>(point.y - 1)[point.x] = -1;
                            }
                            if (point.x > 0 && point.y > 0 && labels.ptr<int>(point.y - 1)[point.x - 1] == 0)
                            {
                                stack.emplace(point.x - 1, point.y - 1); // left top
                                labels.ptr<int>(point.y - 1)[point.x - 1] = -1;
                            }
                            if (point.y > 0 && point.x < image.rows - 1 &&
                                labels.ptr<int>(point.y - 1)[point.x + 1] == 0)
                            {
                                stack.emplace(point.x + 1, point.y - 1); // right top
                                labels.ptr<int>(point.y - 1)[point.x + 1] = -1;
                            }
                            if (point.x > 0 && labels.ptr<int>(point.y)[point.x - 1] == 0)
                            {
                                stack.emplace(point.x - 1, point.y); // left
                                labels.ptr<int>(point.y)[point.x - 1] = -1;
                            }
                            if (point.x < image.rows - 1 && labels.ptr<int>(point.y)[point.x + 1] == 0)
                            {
                                stack.emplace(point.x + 1, point.y); // right
                                labels.ptr<int>(point.y)[point.x + 1] = -1;
                            }
                            if (point.y < image.cols - 1 && labels.ptr<int>(point.y + 1)[point.x] == 0)
                            {
                                stack.emplace(point.x, point.y + 1); // bottom
                                labels.ptr<int>(point.y + 1)[point.x] = -1;
                            }
                            if (point.x > 0 && point.y < image.cols - 1 && labels.ptr<int>(point.y + 1)[point.x - 1] == 0)
                            {
                                stack.emplace(point.x - 1, point.y + 1); // left bottom
                                labels.ptr<int>(point.y + 1)[point.x - 1] = -1;
                            }
                            if (point.y < image.cols - 1 && point.x < image.rows - 1 &&
                                labels.ptr<int>(point.y + 1)[point.x + 1] == 0)
                            {
                                stack.emplace(point.x + 1, point.y + 1); // right bottom
                                labels.ptr<int>(point.y + 1)[point.x + 1] = -1;
                            }
                        }
                    }

                    result.push_back(component);
                    ++n;
                }
            }
        }
        return {labels, result};
    }
}