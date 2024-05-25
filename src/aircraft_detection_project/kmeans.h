#pragma once 
#include "ipaConfig.h"
#include "ucasConfig.h"

cv::Mat kmeansBySize(const std::vector<cv::Mat>& extracted_templates, const int K);


cv::Mat kmeansByIntensity(const std::vector<cv::Mat>& extracted_templates, const int K_clusters);

void kmeansByIntensity1();

void reshape2sameDim1(const int num_clusters_by_size, const int num_clusters_by_intensity);




void reshape2sameDim(std::vector<cv::Mat>& cluestred_imgs_by_intensity);
