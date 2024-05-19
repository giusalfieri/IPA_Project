#include "utils.h"



double degrees2rad(double degrees)
{
	return (degrees / 180) * ipa::PI;
}


double rad2degrees(double radians)
{
	return (radians / ipa::PI) * 180;
}

bool sortByDescendingArea(object& first, object& second)
{
	return cv::contourArea(first) > contourArea(second);
}

cv::Rect Yolo2BRect(const cv::Mat& img, double x_center, double y_center, double width, double height)
{
	// Convert normalized coordinates to pixel values
	int x_center_px = ucas::round<float>(x_center * img.cols);
	int y_center_px = ucas::round<float>(y_center * img.rows);
	int width_px = ucas::round<float>(width * img.cols);
	int height_px = ucas::round<float>(height * img.rows);

	// Calculate top left corner of bounding box
	int x = x_center_px - width_px / 2;
	int y = y_center_px - height_px / 2;

	return cv::Rect(x, y, width_px, height_px);
}


cv::Mat getRotationROI(cv::Mat& img, cv::Rect& roi) {

	cv::Mat rotation_roi;

	// Check if the ROI is within the image boundaries
	if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= img.cols && roi.y + roi.height <= img.rows)
		rotation_roi = img(roi);
	else
	{
		// If the ROI is out of the image boundaries, pad the image
		cv::Mat padding_clone;
		int top = std::max(-roi.y, 0);
		int bottom = std::max(roi.y + roi.height - img.rows, 0);
		int left = std::max(-roi.x, 0);
		int right = std::max(roi.x + roi.width - img.cols, 0);

		cv::copyMakeBorder(img, padding_clone, top, bottom, left, right, cv::BORDER_REFLECT, 0);

		roi.x += left;
		roi.y += top;

		// Apply the ROI to the padded image
		rotation_roi = padding_clone(roi);
	}


	return rotation_roi;
}


// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
cv::Mat rotate90(cv::Mat img, int step)
{
	cv::Mat img_rot;

	// adjust step in case it is negative
	if (step < 0)
		step = -step;
	// adjust step in case it exceeds 4
	step = step % 4;

	// no rotation
	if (step == 0)
		img_rot = img;
	// 90° CW rotation
	else if (step == 1)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 1);
	}
	// 180° CW rotation
	else if (step == 2)
		cv::flip(img, img_rot, -1);
	// 270° CW rotation
	else if (step == 3)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 0);
	}

	return img_rot;
}


std::filesystem::path createDirectoryInParent(const std::string& folder_path, const std::string& directory_name)
{
	// Ottieni il percorso della directory padre
	std::filesystem::path dataset_path(folder_path);
	std::filesystem::path parent_path = dataset_path.parent_path();

	// Crea un nuovo percorso per la nuova directory
	std::filesystem::path new_dir_path = parent_path / directory_name;

	if (!std::filesystem::exists(new_dir_path))
		std::filesystem::create_directory(new_dir_path);


	return new_dir_path;
}
