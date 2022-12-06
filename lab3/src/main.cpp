#include <iostream>

#include <opencv2/highgui.hpp>

#include "median_filter.hpp"

using namespace tud::cvlabs;

int main(int argc, char **argv)
{
    cv::Mat image;
    int taskId;

    if (argc == 2)
    {
        image = cv::imread(cv::samples::findFile("lena.jpg"));
    }
    else if (argc == 3)
    {
        image = cv::imread(argv[2], cv::IMREAD_COLOR);
    }
    else
    {
        std::cerr << "Enter task id (1-3) and/or filepath" << std::endl;
        return -1;
    }

    if (!image.data)
    {
        std::cerr << "Image filepath is incorrect" << std::endl;
        return -1;
    }

    // image.convertTo(image, CV_8UC1);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    std::cout << "\nImage type: " << cv::typeToString(image.type())
              << "\nImage depth: " << cv::depthToString(image.depth())
              << std::endl;

    switch (atoi(argv[1]))
    {
    case 1:
    {
        // cv::imshow("Image", image);
        // cv::Mat reference;
        // cv::medianBlur(image, reference, 3);
        // cv::imshow("ref", reference);

        auto med3x3 = medianFilter(image, 3, 3);
        cv::imshow("Median 3x3", med3x3);
        cv::waitKey();
    }
    break;
    case 2:
        break;
    case 3:
        break;
    default:
        std::cerr << "Unsupported task id" << std::endl;
        return -1;
    }

    return 0;
}