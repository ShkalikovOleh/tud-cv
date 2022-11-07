#include <iostream>
// #include <execution>

#include <opencv2/highgui.hpp>

#include "gamma_correction.hpp"

namespace tud::cvlabs
{
    auto applyGammaCorrection(const cv::Mat &image, double gamma)
    {
        cv::Mat lut(1, 256, CV_8U);
        uchar *lutPtr = lut.ptr();

        for (int i = 0; i < 256; ++i)
        {
            lutPtr[i] = 255 * std::pow(i / 255., gamma);
        }

        cv::Mat result;
        cv::LUT(image, lut, result);

        return std::tuple(result, lut);
    }

    const cv::String curveWindName = "Gamma curve";
    const uint HW = 512;

    void onMouseCallback(int event, int x, int y, int flags, void *data)
    {
        auto imagePtr = static_cast<cv::Mat *>(data);

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            auto norm = static_cast<double>(HW - 1);
            auto logY = std::log((HW - 1 - y) / norm);
            auto logX = std::log(x / norm);
            auto gamma = logY / logX;

            std::cout << "X: " << x
                      << " Y: " << y << std::endl
                      << "Gamma: " << gamma << std::endl;

            auto [result, lut] = applyGammaCorrection(*imagePtr, gamma);
            imshow("Gamma Corrected Image", result);

            std::vector<cv::Point> gammaCurvePoints(256);
            auto lutPtr = lut.ptr();
            auto scaler = HW / 256.;
            for (int i = 0; i < 256; ++i)
            {
                gammaCurvePoints[i] = cv::Point(scaler * i, scaler * (255 - lutPtr[i]));
            }

            cv::Mat curveImg = cv::Mat::zeros(HW, HW, CV_8U);
            cv::polylines(curveImg, gammaCurvePoints, false, cv::Scalar(255));
            cv::imshow(curveWindName, curveImg);
        }
    }

    void gammaCurveMain(cv::Mat &image)
    {
        cv::imshow("Original image", image);

        cv::namedWindow(curveWindName, cv::WindowFlags::WINDOW_AUTOSIZE);
        cv::imshow(curveWindName, cv::Mat::zeros(HW, HW, CV_8U));

        cv::setMouseCallback(curveWindName, onMouseCallback, &image);

        cv::waitKey();
    }
}
