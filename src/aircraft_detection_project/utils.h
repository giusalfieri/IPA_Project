#pragma once
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>



typedef std::vector <cv::Point>  object;

bool sortByDescendingArea(const object& first, const object& second);

double degrees2rad(double degrees);

double rad2degrees(double radians);

std::filesystem::path createDirectory(const std::filesystem::path& folder_path, const std::string& directory_name);

void globFiles(const std::string& directory, const std::string& pattern, std::vector<std::string>& file_paths);

void readImages(const std::vector<std::string>& img_paths, std::vector<cv::Mat>& images, int flags = cv::IMREAD_UNCHANGED);

cv::Mat rotate90(cv::Mat img, int step);

void processYoloLabels(const std::string& filePath, const cv::Mat& img, std::vector<cv::Rect>& yolo_boxes);

cv::Rect Yolo2BRect(const cv::Mat& img, double x_center, double y_center, double width, double height);

bool isRoiInImage(const cv::Rect& roi, int width=4800, int height=2703);

std::vector<cv::Rect> readYoloBoxes(const std::filesystem::path& file_path, const cv::Mat& img);

std::ofstream openFile(const std::string& filename);

std::vector<cv::Rect> generateRoisFromPoints(const std::vector<cv::Point>& points, const std::vector<cv::Size>& roi_sizes);

cv::Size calculateAvgDims(const std::filesystem::path& directory_path);

void reshape2sameDim(std::vector<cv::Mat>& clustered_imgs_by_intensity, const cv::Size& avg_dim);

std::vector<cv::Point> filterPointsByMinDistance(std::vector<cv::Point>& points, double min_distance);

void listDirectories(const std::filesystem::path& directory_path, std::vector<std::string>& final_paths);

void imshow(const std::string& win_name, cv::InputArray arr, bool wait = true, float scale = 1.0);

int bitdepth(int ocv_depth);