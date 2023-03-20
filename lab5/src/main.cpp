#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

#include <opencv2/highgui.hpp>

#include "min_cut.hpp"

using namespace tud::cvlabs;

void pixelClassification(const cv::Mat &image)
{
    cfunc c = [](int x, int y, cv::Vec3b color)
    {
        cv::Vec3i yellow(8, 208, 231);
        float threshold = 50;

        int l1DiffYellow = 0;
        for (int i = 0; i < 3; ++i)
        {
            auto ci = static_cast<int>(color[i]);
            auto yi = yellow[i];
            l1DiffYellow += (ci - yi) * (ci - yi);
        }

        float yc = std::sqrt(l1DiffYellow) - threshold;
        return yc;
    };

    auto ccZero = [](int x1, int y1, cv::Vec3b color1, int x2, int y2, cv::Vec3b color2)
    {
        return 0;
    };

    auto mask = min_cut_classify(image, c, ccZero);

    cv::imshow("Image", image);
    cv::imshow("Simple classification", mask * 127);

    cv::namedWindow("Smooth classification", cv::WINDOW_AUTOSIZE);

    cv::TrackbarCallback trackbarCallback = [](int ccVal, void *data)
    {
        auto pair = static_cast<std::pair<cv::Mat, cfunc> *>(data);
        auto image = pair->first;
        auto c = pair->second;

        auto cc = [ccVal](int x1, int y1, cv::Vec3b color1, int x2, int y2, cv::Vec3b color2)
        {
            return ccVal;
        };

        auto smoothMask = min_cut_classify(image, c, cc);

        cv::imshow("Smooth classification", smoothMask * 127);
    };

    auto args = std::make_pair(image, c);
    cv::createTrackbar("CC value", "Smooth classification", nullptr, 20, trackbarCallback, &args);
    cv::setTrackbarPos("CC value", "Smooth classification", 10);

    cv::waitKey();
}

void min_cut_max_flow_task1(float w)
{
    Graph net{4};
    net.addEdge(0, 2, 3);
    net.addEdge(1, 2, w);
    net.addEdge(1, 3, 2);

    net.addEdge(2, 1, w);
    net.addEdge(2, 0, 0);
    net.addEdge(3, 1, 0);

    auto net2 = net;
    auto res = min_cut_edmonds_karp(net, 0, 3);

    std::cout << "Double directed pipe (w = " << w << ")" << std::endl;
    std::cout << "S = { ";
    for (auto &&v : res)
        std::cout << " " << v << " ";
    std::cout << "}" << std::endl;

    net2.getEdge(2, 1)->w = 0;
    auto res2 = min_cut_edmonds_karp(net2, 0, 3);

    std::cout << "One directed pipe (w = " << w << ")" << std::endl;
    std::cout << "S = { ";
    for (auto &&v : res2)
        std::cout << " " << v << " ";
    std::cout << "}" << std::endl;
}

int main(int argc, char **argv)
{
    cv::Mat image;

    if (argc == 1)
    {
        image = cv::imread(cv::samples::findFile("lena.jpg"), cv::ImreadModes::IMREAD_COLOR);
    }
    else if (argc == 2)
    {
        image = cv::imread(argv[1], cv::ImreadModes::IMREAD_COLOR);
    }
    else
    {
        std::cerr << "Enter filepath or nothing" << std::endl;
        return -1;
    }

    if (!image.data)
    {
        std::cerr << "Image filepath is incorrect" << std::endl;
        return -1;
    }

    std::cout << "\nImage type: " << cv::typeToString(image.type())
              << "\nImage depth: " << cv::depthToString(image.depth())
              << std::endl;

    min_cut_max_flow_task1(1);
    min_cut_max_flow_task1(3);
    min_cut_max_flow_task1(5);

    pixelClassification(image);

    return 0;
}