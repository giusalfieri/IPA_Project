#pragma once

#include <opencv2/core/mat.hpp>


cv::Mat eigenPlanes(const std::vector<cv::Mat>& vec, cv::Size img_dims);