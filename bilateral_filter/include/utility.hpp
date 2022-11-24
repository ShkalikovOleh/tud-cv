#pragma once

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace tud
{
    template <typename Func, typename... Args>
    auto measure_time(std::string name, Func f, Args &&...args)
    {
        auto ts = cv::getTickCount();

        auto res = f(std::forward<Args>(args)...);

        auto te = cv::getTickCount();
        std::cout << name + " time: " << (te - ts) / cv::getTickFrequency() << std::endl;

        return res;
    }

    template <typename Func, typename... Args>
    void measure_time_void(std::string name, Func f, Args &&...args)
    {
        auto ts = cv::getTickCount();

        f(std::forward<Args>(args)...);

        auto te = cv::getTickCount();
        std::cout << name + " time: " << (te - ts) / cv::getTickFrequency() << std::endl;
    }
}