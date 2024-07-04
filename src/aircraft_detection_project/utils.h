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

std::vector<cv::Rect> readYoloBoxes(const std::filesystem::path& file_path);

std::ofstream openFile(const std::string& filename);

std::vector<cv::Rect> generateRoisFromPoints(const std::vector<cv::Point>& points, const std::array<cv::Size, 5>& roi_sizes);

cv::Size calculateAvgDims(const std::filesystem::path& directory_path);

void reshape2sameDim(std::vector<cv::Mat>& clustered_imgs_by_intensity, const cv::Size& avg_dim);

std::vector<cv::Point> filterPointsByMinDistance(std::vector<cv::Point>& points, double min_distance);

//global variable. See https://stackoverflow.com/questions/45710667/defining-global-constants-in-c17
inline const std::array<cv::Size, 5> roi_sizes = { cv::Size(172, 197),
									               cv::Size(276, 299),
	                                               cv::Size(427, 458),
									               cv::Size(602, 647),
									               cv::Size(908, 989) };

void listDirectories(const std::filesystem::path& directory_path, std::vector<std::string>& final_paths);

void drawRectangles(cv::Mat& src_img_drawing, const std::vector<std::pair<cv::Rect, std::string>>& rois_labels_for_points_inside_yolo, const cv::Scalar& color = cv::Scalar(255, 0, 255), int thickness = 2);

void imshow(const std::string winname, cv::InputArray arr, bool wait = true, float scale = 1.0);

int bitdepth(int ocv_depth);