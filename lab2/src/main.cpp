#include <iostream>

#include <opencv2/highgui.hpp>

#include "gamma_correction.hpp"
#include "sobel_filter.hpp"
#include "hist_equalization.hpp"

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

    std::cout << "\nImage type: " << cv::typeToString(image.type())
              << "\nImage depth: " << cv::depthToString(image.depth())
              << std::endl;

    switch (atoi(argv[1]))
    {
    case 1:
        gammaCurveMain(image);
        break;
    case 2:
        sobelMain(image);
        break;
    case 3:
        histEqualMain(image);
        break;
    default:
        std::cerr << "Unsupported task id" << std::endl;
        return -1;
    }

    return 0;
}