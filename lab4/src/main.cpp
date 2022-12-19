#include <iostream>

#include <opencv2/highgui.hpp>

#include "feature_detection.hpp"

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

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::ColorConversionCodes::COLOR_BGR2GRAY);

    int thrPercent = 75;
    cv::namedWindow("Corners", cv::WINDOW_AUTOSIZE);

    cv::TrackbarCallback trackbarCallback = [](int thrPercent, void *data)
    {
        double min, max;
        auto grayImage = *static_cast<cv::Mat *>(data);

        auto R = computeR(grayImage);
        cv::minMaxLoc(R, &min, &max, nullptr, nullptr);
        std::cout << "min: " << min << " max: " << max << std::endl;

        float threshold = (max - min) * thrPercent / 100. + min;
        auto corners = detectCornersHariss(grayImage, threshold);

        auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(corners, corners, kernel, cv::Point(-1, -1), 3);
        cv::cvtColor(grayImage, grayImage, cv::ColorConversionCodes::COLOR_GRAY2BGR);

        std::vector<cv::Mat> channels;
        cv::split(grayImage, channels);
        channels[0] = channels[0];
        channels[1] = channels[1];
        channels[2] = channels[2] + corners;
        cv::merge(channels, grayImage);

        cv::imshow("Corners", grayImage);
    };

    cv::createTrackbar("Threshold %", "Corners", &thrPercent, 100, trackbarCallback, &grayImage);
    cv::waitKey();

    return 0;
}