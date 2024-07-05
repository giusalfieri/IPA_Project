#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


std::vector<std::vector<float>> hog_features_extraction(const std::vector<cv::Rect>& rois, const cv::Mat& image);

void writeHogFeaturesToCsv(const std::vector<std::vector<float>>& hog_features, const std::string& filename);