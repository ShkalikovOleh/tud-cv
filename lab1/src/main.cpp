#include <iostream>
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "local_max.hpp"

using namespace tud::cvlabs;

cv::Mat markMax(const cv::Mat &image, const cv::Mat &maxMask)
{
    cv::Mat markedImage;
    cv::cvtColor(image, markedImage, cv::COLOR_GRAY2RGB); // convert from grayscale to rgb

    std::vector<cv::Mat> channels;

    cv::split(markedImage, channels);     // split by channels
    channels[2] = channels[2] + maxMask;  // sum is clipped (>255 -> 255)
    channels[1] = channels[1] & ~maxMask; // zero green channel for loc max
    channels[0] = channels[0] & ~maxMask; // zero blue channel for loc max

    cv::merge(channels, markedImage); // merge channels

    return markedImage;
}

void show_result(const cv::Mat &image, const cv::Mat &neigh_mask,
                 std::function<cv::Mat(const cv::Mat &, const cv::Mat &)> func,
                 const cv::String &title)
{
    auto start = cv::getTickCount(); // time measurment

    auto res = func(image, neigh_mask);

    auto stop = cv::getTickCount();
    auto time = (stop - start) / cv::getTickFrequency(); // we have to divide because ticks is not a seconds
    std::cout << title + ": " << time << "s" << std::endl;

    auto markedImage = markMax(image, res);

    const auto windowName = "Local Maxima " + title;
    cv::namedWindow(windowName, cv::WindowFlags::WINDOW_AUTOSIZE);
    cv::imshow(windowName, markedImage);
}

// To point which pixel we recognize as a neighbor we will use mask of neighbours
cv::Mat getNeighborhoodMask(int neighboors)
{
    auto shapeType = neighboors == 8 ? cv::MorphShapes::MORPH_RECT : cv::MorphShapes::MORPH_CROSS;

    auto mask = cv::getStructuringElement(shapeType, cv::Size(3, 3));
    mask.at<uchar>(1, 1) = 0;

    return mask;
}

int main(int argc, char **argv)
{
    cv::Mat image, mask;

    /* Loading image from file
    // User are supposed to specify a path to file and number of neighbors (4 or 8)
    as input parameter */
    if (argc == 3)
    {
        // read file in grayscale (because of the task)
        image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

        if (!image.data)
        {
            std::cerr << "Image filepath is incorrect" << std::endl;
            return -1;
        }

        std::cout << "Path to image: " << argv[1]
                  << "\nImage type: " << cv::typeToString(image.type())
                  << "\nImage depth: " << cv::depthToString(image.depth())
                  << std::endl;

        int neighType = atoi(argv[2]);
        if (neighType != 4 && neighType != 8)
        {
            std::cerr << "Number of neighboors is unsupported" << std::endl;
            return -1;
        }

        mask = getNeighborhoodMask(neighType);
        std::cout << "Neighborhood mask\n"
                  << mask << std::endl;
    }
    else
    {
        std::cerr << "Please specify the filepath and number of neighboors" << std::endl;
        return -1;
    }

    show_result(image, mask, localMaxNaive, "Naive");
    show_result(image, mask, localMaxNaiveParallel, "Parallel");
    show_result(image, mask, localMaxDilate, "Dilate");

    cv::waitKey();

    return 0;
}