#pragma once

#include <filesystem>
#include <opencv2/core/mat.hpp>


cv::Mat kmeansBySize(const std::vector<cv::Mat>& extracted_templates, int K);

cv::Mat kmeansByIntensity(const std::vector<cv::Mat>& extracted_templates, int K_clusters);

void saveClusteredImages(const std::vector<cv::Mat>& images, const std::vector<std::string>& image_paths, const cv::Mat& labels, const std::vector<std::filesystem::path>& cluster_paths);