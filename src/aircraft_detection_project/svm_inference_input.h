#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>



void classifyPointsByYoloBoxes(const std::vector<cv::Rect>& yolo_boxes,const std::vector<cv::Point>& max_corr_points,std::vector<cv::Point>& max_corr_tp,std::vector<cv::Point>& max_corr_fp);

std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> associateYoloBoxesWithRois(const std::vector<cv::Rect>& yolo_boxes,const std::vector<cv::Point>& max_corr_tp);

std::vector<std::pair<cv::Rect, std::string>> labelRoisBasedOnIouWithYoloBoxes(const std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>>& yoloBox_roi_pairs,double iou_threshold);

std::vector<std::pair<cv::Rect, std::string>> generateRoisWithLabelsFromPoints_Outside_Yolo(const std::vector<cv::Point>& max_corr_fp);

std::vector<std::pair<cv::Rect, std::string>> concatenateRoiLabelPairs(const std::vector<std::pair<cv::Rect, std::string>>& roi_labels_pairs_point_in_yolo, const std::vector<std::pair<cv::Rect, std::string>>& roi_labels_pairs_point_outside_yolo);

std::vector<cv::Rect> extractRois(const std::vector<std::pair<cv::Rect, std::string>>& roi_labels_pairs);

void saveRoiLabelPairsToFile(const std::vector<std::pair<cv::Rect, std::string>>& roi_labels, const std::string& filename);