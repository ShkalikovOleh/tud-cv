#include <functional>

#include <opencv2/opencv.hpp>

#include "bilateral_filter.hpp"
#include "utility.hpp"

using namespace tud;

float unnormed_normal_pdf(float sigma, float x)
{
    auto sigma_sq_n = -0.5 / (sigma * sigma);
    return std::exp(x * x * sigma_sq_n);
}

float normal_pdf(float sigma, float x)
{
    const float PI = 3.141592653589793;
    auto norm = 1 / (std::sqrt(2 * PI) * sigma);
    return norm * unnormed_normal_pdf(sigma, x);
}

float dc(float sigma, float x)
{
    auto sigma_sq = sigma * sigma;
    return 1 / (1 + (x * x) / sigma_sq);
}

int main(int argc, char **argv)
{
    auto path = cv::samples::findFile("fruits.jpg");
    auto image = cv::imread(path);
    cv::imshow("Image", image);

    const float sigma_s = 10;
    const float sigma_c = 10;
    const int N = 3;

    cv::Mat gaussBlur;
    cv::GaussianBlur(image, gaussBlur, cv::Size(2 * N + 1, 2 * N + 1), sigma_s);
    imshow("Gauss Blur", gaussBlur);

    auto ds_f = std::bind(normal_pdf, sigma_s, std::placeholders::_1);
    auto dc_f = std::bind(dc, sigma_c, std::placeholders::_1);
    // auto ds_f = std::bind(unnormed_normal_pdf, sigma_s, std::placeholders::_1);
    // auto dc_f = std::bind(unnormed_normal_pdf, sigma_c, std::placeholders::_1);

    auto filtered = measure_time("Our solution", bilateral_filter, image, 3, ds_f, dc_f);
    cv::imshow("Bilateral filtered image (our solution)", filtered);

    cv::Mat reference;
    measure_time_void("OpenCV version", cv::bilateralFilter, image,
                      reference, N * 2 + 1, sigma_c, sigma_s,
                      cv::BorderTypes::BORDER_DEFAULT);
    cv::imshow("Open CV result", reference);

    cv::waitKey();
    return 0;
}