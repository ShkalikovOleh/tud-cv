#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

#include <opencv2/highgui.hpp>

#include "feature_detection.hpp"
#include "pixel_classification.hpp"

using namespace tud::cvlabs;

void cornerDetection(const cv::Mat &image)
{
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::ColorConversionCodes::COLOR_BGR2GRAY);

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
        cv::dilate(corners, corners, kernel, cv::Point(-1, -1), 3); // to make corner visible on the image
        cv::cvtColor(grayImage, grayImage, cv::ColorConversionCodes::COLOR_GRAY2BGR);

        std::vector<cv::Mat> channels;
        cv::split(grayImage, channels);
        channels[0] = channels[0];
        channels[1] = channels[1];
        channels[2] = channels[2] + corners;
        cv::merge(channels, grayImage);

        cv::imshow("Corners", grayImage);
    };

    cv::createTrackbar("Threshold %", "Corners", nullptr, 100, trackbarCallback, &grayImage);
    cv::setTrackbarPos("Threshold %", "Corners", 75);
    cv::waitKey();
}

void pixelClassification(const cv::Mat &image)
{
    cfunc c = [](int x, int y, cv::Vec3b color)
    {
        cv::Vec3i yellow(8, 208, 231);
        cv::Vec3i white(241, 250, 247);
        float threshold = 50;

        int l1DiffYellow = 0, l1DiffWhite = 0;
        for (int i = 0; i < 3; ++i)
        {
            auto ci = static_cast<int>(color[i]);
            auto yi = yellow[i];
            auto wi = white[i];
            l1DiffYellow += (ci - yi) * (ci - yi);
            l1DiffWhite += (ci - wi) * (ci - wi);
        }

        float yc = std::sqrt(l1DiffYellow) - threshold;
        float wc = std::sqrt(l1DiffWhite) - threshold;

        std::vector<float> c{threshold, yc, wc};
        return c;
    };

    auto ccZero = [](int x1, int y1, cv::Vec3b color1, int x2, int y2, uchar label2, cv::Vec3b color2)
    {
        std::vector<float> cc(3);
        std::fill(cc.begin(), cc.end(), 0);
        return cc;
    };

    auto mask = classify(image, c, ccZero);

    cv::imshow("Image", image);
    cv::imshow("Simple classification", mask * 127);

    cv::namedWindow("Smooth classification", cv::WINDOW_AUTOSIZE);

    cv::TrackbarCallback trackbarCallback = [](int ccVal, void *data)
    {
        auto pair = static_cast<std::pair<cv::Mat, cfunc> *>(data);
        auto image = pair->first;
        auto c = pair->second;

        auto ccSmooth = [ccVal](int x1, int y1, cv::Vec3b color1, int x2, int y2, uchar label2, cv::Vec3b color2)
        {
            std::vector<float> cc(3);
            std::fill(cc.begin(), cc.end(), ccVal);
            cc[label2] = 0; // don't penaltize for the equal labels
            return cc;
        };

        auto smoothMask = classify(image, c, ccSmooth);

        cv::imshow("Smooth classification", smoothMask * 127);
    };

    auto args = std::make_pair(image, c);
    cv::createTrackbar("CC value", "Smooth classification", nullptr, 20, trackbarCallback, &args);
    cv::setTrackbarPos("CC value", "Smooth classification", 10);

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