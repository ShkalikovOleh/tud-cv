#include "feature_detection.hpp"
#include "tuple"

namespace tud::cvlabs
{
    cv::Mat nms(const cv::Mat &image)
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
        cv::Mat lmax, result;

        cv::dilate(image, lmax, kernel); // local max in result

        // cv::bitwise_and(lmax, lmax, result, image == result);
        cv::copyTo(lmax, result, lmax == image);

        return result;
    }

    StructureTensorComponents computeStructureTensor(const cv::Mat &image)
    {
        // derivatives
        double scale = 1 / (4 * 255.); // to match OpenCV result and avoid large numbers
        cv::Mat dX2, dY2, dXY;
        cv::Sobel(image, dX2, CV_32F, 1, 0, 3, scale);
        cv::Sobel(image, dY2, CV_32F, 0, 1, 3, scale);

        // squared derivatives
        cv::multiply(dX2, dY2, dXY);
        cv::multiply(dX2, dX2, dX2);
        cv::multiply(dY2, dY2, dY2);

        // structure tensor components
        cv::Mat H11, H22, H12;
        cv::boxFilter(dX2, H11, CV_32F, cv::Size(3, 3)); // the same as filter2D with 1/9 kernel's values
        cv::boxFilter(dY2, H22, CV_32F, cv::Size(3, 3));
        cv::boxFilter(dXY, H12, CV_32F, cv::Size(3, 3));

        return std::make_tuple(H11, H12, H22);
    }

    cv::Mat computeLowerEigVals(const cv::Mat &image)
    {
        auto [H11, H12, H22] = computeStructureTensor(image);

        // min eigvals
        auto b = H11 + H22; // minus b
        cv::Mat bsq;
        cv::multiply(b, b, bsq);

        cv::Mat c1, c2;
        cv::multiply(H11, H22, c1);
        cv::multiply(H12, H12, c2);
        auto c = c1 - c2;

        cv::Mat D = bsq - 4 * c;
        cv::sqrt(D, D);

        cv::Mat result = (b - D) / 2;
        return result;
    }

    // actually really similar to the min eigvals in applications but with less computations
    cv::Mat computeR(const cv::Mat &image, float k)
    {
        auto [H11, H12, H22] = computeStructureTensor(image);

        cv::Mat det1, det2;
        cv::multiply(H11, H22, det1);
        cv::multiply(H12, H12, det2);
        auto det = det1 - det2;

        cv::Mat trace_sq, trace = H11 + H22;
        cv::multiply(trace, trace, trace_sq);

        return det - k * trace_sq;
    }

    cv::Mat detectCornersHariss(const cv::Mat &image, float threshold, float k)
    {
        auto R = computeR(image, k);
        cv::threshold(R, R, threshold, 255, cv::ThresholdTypes::THRESH_TOZERO);
        auto corners = nms(R);
        return corners > 0;
    }
}