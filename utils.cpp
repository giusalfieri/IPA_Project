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


std::filesystem::path createDirectory(const std::filesystem::path& folder_path, const std::string& directory_name)
{
	

	// Crea un nuovo percorso per la nuova directory
	std::filesystem::path new_dir_path = folder_path / directory_name;

	if (!std::filesystem::exists(new_dir_path))
		std::filesystem::create_directory(new_dir_path);


	return new_dir_path;
}