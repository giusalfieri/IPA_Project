#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>



void readImageAndAnnotations(const std::string& testing_img_id, cv::Mat& src_img, cv::Mat& src_img_gray, std::vector<cv::Rect>& yolo_boxes);

std::vector<std::pair<cv::Rect, std::string>> classifyAndLabelRois(const std::vector<cv::Rect>& yolo_boxes, const std::vector<cv::Point>& max_corr_points);

void saveResults(const std::string& testing_img_id, const std::vector<std::pair<cv::Rect, std::string>>& rois_label_pairs_all_points, const cv::Mat& src_img_gray,
    const std::filesystem::path& output_dir);

void drawAndSaveResults(const cv::Mat& src_img, const std::vector<std::pair<cv::Rect, std::string>>& rois_label_pairs_all_points, const std::filesystem::path& output_dir);