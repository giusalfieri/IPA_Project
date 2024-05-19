#pragma once

#include "ipaConfig.h"
#include "ucasConfig.h"
#include <filesystem>


typedef std::vector <cv::Point>  object;


bool sortByDescendingArea(object& first, object& second);

double degrees2rad(double degrees);

double rad2degrees(double radians);

cv::Rect Yolo2BRect(const cv::Mat& input_img, double x_center, double y_center, double width, double height);

cv::Mat getRotationROI(cv::Mat& img, cv::Rect& roi);



std::filesystem::path createDirectory(const std::filesystem::path& folder_path, const std::string& directory_name);

// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
cv::Mat rotate90(cv::Mat img, int step);
