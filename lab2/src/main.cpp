#include <iostream>
#include <algorithm>

#include <opencv2/highgui.hpp>

#include "gamma_correction.hpp"
#include "sobel_filter.hpp"
#include "hist_equalization.hpp"

using namespace tud::cvlabs;

void sobelMain(const cv::Mat &image)
{
    auto result = applySobel(image);

    imshow("Our Sobel Applied", result);
    imshow("Original Image", image);

    cv::Mat reference;
    cv::Sobel(image, reference, -1, 1, 0);
    imshow("Reference Sobel", reference);

    cv::waitKey();
}

void showHist(const cv::Mat &hist)
{
    std::vector<uchar> y;

    double max = *std::max_element(hist.begin<double>(), hist.end<double>());
    double norm = (255 - 20) / max;

    std::transform(hist.begin<double>(), hist.end<double>(), std::back_inserter(y),
                   [norm](double h)
                   {
                       return cv::saturate_cast<uchar>(h * norm);
                   });

    cv::Mat result = cv::Mat::zeros(256, 256, CV_8U);
    for (int i = 0; i < 256; ++i)
    {
        cv::rectangle(result, cv::Rect(i, 255 - y[i], 1, y[i]), 255);
    }

    imshow("Histogram", result);
}

void histEqualMain(const cv::Mat &image)
{
    cv::Mat grayImg;
    cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

    auto hist = calculateHistogram(grayImg);
    auto result = applyEqualization(grayImg, hist);

    showHist(hist);
    imshow("Our Hist Equalization", result);
    imshow("Original Image", grayImg);

    cv::Mat reference;
    cv::equalizeHist(grayImg, reference);
    imshow("Reference Hist Equalization", reference);

    cv::waitKey();
}

int main(int argc, char **argv)
{
    cv::Mat image;

    if (argc == 2)
    {
        image = cv::imread(argv[1], cv::IMREAD_COLOR);
    }
    else
    {
        image = cv::imread(cv::samples::findFile("lena.jpg"));
    }

    if (!image.data)
    {
        std::cerr << "Image filepath is incorrect" << std::endl;
        return -1;
    }

    std::cout << "\nImage type: " << cv::typeToString(image.type())
              << "\nImage depth: " << cv::depthToString(image.depth())
              << std::endl;

    gammaCurveMain(image);
    sobelMain(image);
    histEqualMain(image);

    return 0;
}