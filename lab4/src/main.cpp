#include <iostream>

#include <opencv2/highgui.hpp>
#include <algorithm>

#include "feature_detection.hpp"
#include "pixel_classification.hpp"

using namespace tud::cvlabs;

void cornerDetection(const cv::Mat &image)
{
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
}

void pixelClassification(const cv::Mat &image)
{
    auto c = [](int x, int y, cv::Vec3b color)
    {
        // cv::Vec3b yellow(0, 127, 127);
        // cv::Vec3b white(255 / 3, 255 / 3, 255 / 3);

        // float yc = -yellow.dot(color);
        // float wc = -white.dot(color);
        // std::vector<float> c{0, yc, wc};

        std::vector<float> c(3);

        if (color[1] >= 127 && color[2] >= 127)
        {
            if (color[0] <= 127)
            {
                c[1] = -1;
            }
            else
            {
                c[2] = -1;
            }
        }
        else
        {
            c[0] = -1;
        }

        return c;
    };

    auto ccZero = [](int x1, int y1, cv::Vec3b color1, int x2, int y2, cv::Vec3b color2)
    {
        std::vector<float> cc(3);
        std::fill(cc.begin(), cc.end(), 0);
        return cc;
    };

    auto mask = classify(image, c, ccZero);

    cv::imshow("Simple classification", mask * 127);

    cv::waitKey();
}

int main(int argc, char **argv)
{
    cv::Mat image;
    int taskId;

    if (argc == 2)
    {
        image = cv::imread(cv::samples::findFile("lena.png"), cv::ImreadModes::IMREAD_COLOR);
    }
    else if (argc == 3)
    {
        image = cv::imread(argv[2], cv::ImreadModes::IMREAD_COLOR);
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
        cornerDetection(image);
        break;
    case 2:
        pixelClassification(image);
        break;
    default:
        std::cerr << "Unsupported task id" << std::endl;
        return -1;
    }

    return 0;
}