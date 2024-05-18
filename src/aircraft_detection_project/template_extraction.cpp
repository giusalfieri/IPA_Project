#include <iostream>
#include "functions.h"

cv::Mat ipa::faceRectangles(const cv::Mat & frame) 
{
	// find faces
	std::vector < cv::Rect > faces = ipa::faceDetector(frame);

	// for each face found...
	cv::Mat out = frame.clone();
	for(int k=0; k<faces.size(); k++)
		cv::rectangle(out, faces[k], cv::Scalar(0, 0, 255), 2);

	return out;
}
