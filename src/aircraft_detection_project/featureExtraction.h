#include "utils.h"


void featureExtractions(const std::map<std::string, std::vector<cv::Mat>> &samples_for_svm, const std::string &output_directory);
std::vector<float> hogFeatures(const cv::Mat &roi);