#include "utils.h"



constexpr double PI = 3.14159265358979323846; 	


double degrees2rad(double degrees)
{
	return (degrees / 180) * PI;
}

double rad2degrees(double radians)
{
	return (radians / PI) * 180;
}

bool sortByDescendingArea(const object& first, const object& second)
{
	return cv::contourArea(first) > contourArea(second);
}

// utility function that rotates 'img' by step * 90 degree
// step = 0 --> no rotation
// step = 1 --> 90  deg CW rotation
// step = 2 --> 180 deg CW rotation
// step = 3 --> 270 deg CW rotation
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
	// 90 deg CW rotation
	else if (step == 1)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 1);
	}
	// 180 deg CW rotation
	else if (step == 2)
		cv::flip(img, img_rot, -1);
	// 270 deg CW rotation
	else if (step == 3)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 0);
	}

	return img_rot;
}

/**
 * @brief Creates a new directory at the specified path.
 *
 * This function takes a base folder path and a directory name as input,
 * then creates a new directory with the specified name inside the base folder.
 * If the directory already exists, it does nothing and returns the path to the existing directory.
 *
 * @param folder_path The base folder path where the new directory should be created.
 * @param directory_name The name of the new directory to be created.
 * @return std::filesystem::path The full path to the created (or existing) directory.
 */
std::filesystem::path createDirectory(const std::filesystem::path& folder_path, const std::string& directory_name)
{
	std::filesystem::path new_dir_path = folder_path / directory_name;

	if (!std::filesystem::exists(new_dir_path))
		std::filesystem::create_directory(new_dir_path);

	return new_dir_path;
}

/**
 * @brief Retrieves file paths matching a specified pattern within a given directory.
 *
 * Constructs a search pattern from the directory and file name pattern,
 * then uses OpenCV's `cv::glob` to find matching files.
 *
 * @param directory The directory to search in.
 * @param pattern The file name pattern to match (e.g., "*.jpg").
 * @param file_paths A vector to store the paths of matching files.
 */
void globFiles(const std::string& directory, const std::string& pattern, std::vector<std::string>& file_paths)
{
	const std::string full_pattern = directory + "/" + pattern;
	cv::glob(full_pattern, file_paths);
}


/**
 * @brief Reads images from the given file paths and stores them in a vector.
 *
 * This function iterates over a list of image file paths, reads each image using OpenCV's `imread` function
 * with the specified flags, and stores the successfully read images in the provided vector.
 *
 * @param img_paths A vector of strings, each representing a path to an image file.
 * @param images A vector of `cv::Mat` objects where the successfully read images will be stored.
 * @param flags Flags for reading the images, which are passed to `cv::imread`. These flags determine the
 *              color type and depth of the loaded image. Common values include `cv::IMREAD_COLOR`, `cv::IMREAD_GRAYSCALE`, etc.
 */
void readImages(const std::vector<std::string>& img_paths, std::vector<cv::Mat>& images, int flags)
{
	for (const auto& path : img_paths)
	{
		cv::Mat img = cv::imread(path, flags);
		if (img.data)
			images.push_back(img);
	}
}



void processYoloLabels(const std::string& filePath,const cv::Mat& img, std::vector<cv::Rect>& yolo_boxes)
{
	std::ifstream file(filePath);

	// Reading lines from a .txt file.
	// The data on each line is expected to be in the YOLO label format: 
	// <class_id> <center_x> <center_y> <width> <height> 
	// 
	// class_id --> the class label of the object.
	// center_x --> the normalized x coordinate of the bounding box center.
	// center_y --> the normalized y coordinate of the bounding box center.
	// width    --> the normalized width   of the bounding box.
	// height   --> the normalized height  of the bounding box.
	if (file.is_open())
	{
		std::string line;
		while (std::getline(file, line))
		{
			std::istringstream line_stream(line);
			int class_id;
			double center_x, center_y, width, height;
			if (!(line_stream >> class_id >> center_x >> center_y >> width >> height))
			{
				std::cerr << "Error: could not read line '" << line << "'\n";
				continue; // skip this line and continue with the next line
			}
			cv::Rect rect = Yolo2BRect(img, center_x, center_y, width, height);
			if (rect.empty())
			{
				std::cerr << "Error: Bounding box falls beyond the image boundaries.\n";
				continue; // skip this line and continue with the next line
			}
			yolo_boxes.push_back(rect);
		}
	}
	else std::cerr << "Error: could not open file\n";

}


cv::Rect Yolo2BRect(const cv::Mat& img, double x_center, double y_center, double width, double height)
{
	// Convert normalized, [0, 1], coordinates to pixel values
	const int x_center_px = static_cast<int>(std::round(x_center * img.cols));
	const int y_center_px = static_cast<int>(std::round(y_center * img.rows));
	int width_px = static_cast<int>(std::round(width * img.cols));
	int height_px = static_cast<int>(std::round(height * img.rows));

	// Calculate top left corner of bounding box
	int x = x_center_px - width_px / 2;
	int y = y_center_px - height_px / 2;

	// Check if the bounding box falls beyond the image boundaries: if so an empty cv::Rect is returned
	if (x < 0 || y < 0 || x + width_px > img.cols || y + height_px > img.rows)
	{
		std::cerr << "Bounding box falls beyond the image boundaries: an empty cv::Rect will be returned\n";
		return {};
	}

	return { x, y, width_px, height_px };
}


/**
 * @brief Checks if a region of interest (ROI) is completely within the boundaries of an image.
 *
 * This function takes a rectangle representing a region of interest (ROI) and the dimensions of an image,
 * and checks if the ROI is entirely contained within the image boundaries. The function returns true if the
 * ROI is completely inside the image, and false otherwise.
 *
 * @param roi The rectangle representing the region of interest (ROI).
 * @param width The width of the image.
 * @param height The height of the image.
 * @return bool True if the ROI is completely within the image boundaries, false otherwise.
 */
bool isRoiInImage(const cv::Rect& roi,int width, int height)
{
	// Create a rectangle that represents the image boundaries
	const cv::Rect image_rect(0, 0, width, height);

	// Check if the roi is completely inside the image
	return (image_rect & roi) == roi;
}



std::vector<cv::Rect> readYoloBoxes(const std::filesystem::path& file_path, const cv::Mat& img)
{
	std::vector<cv::Rect> yolo_boxes;
	std::ifstream infile(file_path);

	if (!infile.is_open())
		throw std::runtime_error("Unable to open file: " + file_path.string());


	std::string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		int class_id;
		float x_center, y_center, width, height;

		// Assuming the format is "class_id x_center y_center width height"
		if (!(iss >> class_id >> x_center >> y_center >> width >> height))
		{
			std::cerr << "Error parsing line: " << line << "\n";
			continue;
		}

		// Convert YOLO format to OpenCV cv::Rect
		int x = static_cast<int>((x_center - width / 2) * img.cols);
		int y = static_cast<int>((y_center - height / 2) * img.rows);
		int w = static_cast<int>(width * img.cols);
		int h = static_cast<int>(height * img.rows);

		yolo_boxes.emplace_back(x, y, w, h);
	}

	infile.close(); //comment this (RAII will close the file automatically)
	return yolo_boxes;
}



// Helper function to open a file
std::ofstream openFile(const std::string& filename)
{
	std::ofstream file(filename);
	if (!file.is_open())
		throw std::runtime_error("Unable to open file: " + filename);

	return file;
}


std::vector<cv::Rect> generateRoisFromPoints(const std::vector<cv::Point>& points, const std::vector<cv::Size>& roi_sizes)
{
	std::vector<cv::Rect> rois;

	for (const auto& point : points)
	{
		for (const auto& roi_size : roi_sizes)
		{
			const int x = point.x - roi_size.width / 2;
			const int y = point.y - roi_size.height / 2;

			if (cv::Rect roi(x, y, roi_size.width, roi_size.height); isRoiInImage(roi))
			{
				rois.push_back(roi);
			}
		}
	}

	return rois;
}

/**
 * @brief Calculates the average dimensions of PNG images in a specified directory.
 *
 * This function scans a given directory for PNG images, calculates the width and height of each image,
 * and then computes the average dimensions (width and height) of all the images found. If no images are found,
 * or if any image fails to load, it throws an exception.
 *
 * @param directory_path The path to the directory containing the PNG images.
 * @return cv::Size A cv::Size object containing the average width and height of the images.
 * @throws std::runtime_error If no images are found in the directory or if an image fails to load.
 */
cv::Size calculateAvgDims(const std::filesystem::path& directory_path)
{
	// Get all the image paths in the directory
	std::vector<std::string> image_paths;
	cv::glob(directory_path.string() + "/*.png", image_paths);

	if (image_paths.empty()) 
		throw std::runtime_error("No images found in the directory.");

	// Calculate the average dimensions of the images in the directory
	int acc_widths = 0;
	int acc_heights = 0;
	//std::vector<cv::Mat> images;
	int num_images = 0;

	for (const auto& image_path : image_paths) 
	{
		cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
		if (img.empty()) 
			throw std::runtime_error("Failed to load image: " + image_path);

		num_images++;
		//images.push_back(img);
		acc_widths += img.cols;
		acc_heights += img.rows;
		
	}


	return {
		static_cast<int>(std::round(static_cast<float>(acc_widths) / num_images)),
		static_cast<int>(std::round(static_cast<float>(acc_heights) / num_images))
	};
}


/**
 * @brief Resizes a collection of images to the same specified dimensions.
 *
 * This function takes a vector of images and a target size, then resizes each image in the vector to the specified dimensions.
 * The resizing is done in place, modifying the original images.
 *
 * @param clustered_imgs_by_intensity A vector of cv::Mat objects representing the images to be resized.
 * @param avg_dim A cv::Size object specifying the target dimensions (width and height) to which each image should be resized.
 */
void reshape2sameDim(std::vector<cv::Mat>& clustered_imgs_by_intensity,const cv::Size& avg_dim)
{
	for (auto& img : clustered_imgs_by_intensity) 
		cv::resize(img, img, avg_dim);
	//cv::resize(img, img, avg_dim, 0, 0, cv::INTER_CUBIC);
}



double euclidean_distance(const cv::Point& a, const cv::Point& b)
{
	return cv::norm(a - b);
}


/**
 * @brief Filters a list of points to ensure a minimum distance between them.
 *
 * This function takes a vector of points and a minimum distance, then filters the points such that no two points
 * in the resulting vector are closer to each other than the specified minimum distance. The points are first sorted
 * by their coordinates for efficient processing.
 *
 * @param points A vector of cv::Point objects representing the points to be filtered.
 * @param min_distance The minimum distance that must be maintained between any two points in the resulting vector.
 * @return std::vector<cv::Point> A vector of cv::Point objects that are filtered to maintain the specified minimum distance.
 */
std::vector<cv::Point> filterPointsByMinDistance(std::vector<cv::Point>& points, double min_distance)
{
	// Sort the points by their coordinates
	std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b)
	{
		return a.x < b.x || (a.x == b.x && a.y < b.y);
	});

	// Select the points that are not too close to each other
	std::vector<cv::Point> selected_points;
	for (const cv::Point& point : points)
	{
		bool too_close = false;

		//Check if the point is too close to any previously selected points
		for (const cv::Point& selected_point : selected_points)
		{
			if (euclidean_distance(point, selected_point) < min_distance)
			{
				too_close = true;
				break;
			}
		}

		// If the point is not too close to any previously selected points, select it
		if (!too_close)
			selected_points.push_back(point);

	}
	return selected_points;
}


/**
 * @brief Recursively lists all leaf directories in a given directory.
 *
 * This function takes a directory path and recursively searches for all leaf directories within it.
 * A leaf directory is defined as a directory that contains no subdirectories. The paths of these leaf
 * directories are stored in the provided vector.
 *
 * @param directory_path The path to the directory to be searched.
 * @param final_paths A vector of strings where the paths of the leaf directories will be stored.
 */
void listDirectories(const std::filesystem::path& directory_path, std::vector<std::string>& final_paths)
{
	bool isLeaf = true;

	for (const auto& entry : std::filesystem::directory_iterator(directory_path))
	{
		if (entry.is_directory())
		{
			isLeaf = false; // Found a subdirectory
			listDirectories(entry.path(), final_paths);
		}
	}

	if (isLeaf)
		final_paths.push_back(directory_path.string());
	
}





/*
********************************************
*        OpenCV UTILITY functions	       *
********************************************
*/

// wrapper that adapts cv::imshow to the current screen
void imshow(const std::string& win_name, cv::InputArray arr, bool wait, float scale)
{
	// create window
	cv::namedWindow(win_name, cv::WINDOW_KEEPRATIO);

	// resize window to fit screen size while maintaining image aspect ratio
	int win_height = arr.size().height, win_width = arr.size().width;

	cv::resizeWindow(win_name, round(win_width * scale), round(win_height * scale));

	// display image
	cv::imshow(win_name, arr);

	// wait for key pressed
	if (wait)
		cv::waitKey(0);
}

// convert OpenCV depth (which is a macro) to the corresponding bit depth
int bitdepth(int ocv_depth)
{
	switch (ocv_depth)
	{
	case CV_8U:  return 8;
	case CV_8S:  return 8;
	case CV_16U: return 16;
	case CV_16S: return 16;
	case CV_32S: return 32;
	case CV_32F: return 32;
	case CV_64F: return 64;
	default:     return -1;
	}
}