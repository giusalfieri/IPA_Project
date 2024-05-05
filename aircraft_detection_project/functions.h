#pragma once

#include "ipaConfig.h"
#include <opencv2/core/core.hpp>

// open namespace "ipa"
namespace ipa
{
	// this is just an example: find all faces in the given image using HaarCascade Face Detection
	cv::Mat faceRectangles(const cv::Mat & frame) ;
}