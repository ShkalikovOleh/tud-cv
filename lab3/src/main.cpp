#include <iostream>

#include <opencv2/highgui.hpp>

#include "median_filter.hpp"
#include "connected_component.hpp"

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

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    std::cout << "\nImage type: " << cv::typeToString(image.type())
              << "\nImage depth: " << cv::depthToString(image.depth())
              << std::endl;

    switch (atoi(argv[1]))
    {
    case 1:
    {
        auto med3x3 = medianFilter(image, 3, 3);
        cv::imshow("Median 3x3", med3x3);
        cv::waitKey();
    }
    break;
    case 2:
    {
        cv::threshold(image, image, 128, 255, cv::ThresholdTypes::THRESH_BINARY);
        auto [labels, components] = floodFilling(image);
        std::cout << "Number of components: " << components.size() << std::endl;

        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

        for (auto &&component : components)
        {
            std::cout << "Number of points: " << component.n_points() << std::endl;
            std::cout << "Bounding box: " << component.bounding_box() << std::endl;
            std::cout << "Centroid: " << component.centroid() << std::endl;
            std::cout << "Eccentricity: " << component.eccentricity() << std::endl;
            std::cout << "Orientation: " << component.orientation() << std::endl;

            cv::rectangle(image, component.bounding_box(), cv::Scalar(0, 0, 255));
        }

        cv::imshow("Binary", image);
        cv::waitKey();
    }
    break;
    case 3:
        break;
    default:
        std::cerr << "Unsupported task id" << std::endl;
        return -1;
    }

    return 0;
}