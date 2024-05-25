#pragma once

#include "ipaConfig.h"
#include "ucasConfig.h"
#include <filesystem>


typedef std::vector <cv::Point>  object;




bool sortByDescendingArea(object& first, object& second);

double degrees2rad(double degrees);

double rad2degrees(double radians);



std::filesystem::path createDirectory(const std::filesystem::path& folder_path, const std::string& directory_name);

void globFiles(const std::string& directory, const std::string& pattern, std::vector<std::string>& file_paths);


void readImages(const std::vector<std::string>& img_paths, std::vector<cv::Mat>& images);

// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
cv::Mat rotate90(cv::Mat img, int step);
