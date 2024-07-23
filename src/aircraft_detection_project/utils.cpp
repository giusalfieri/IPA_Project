#include "utils.h"



constexpr double PI = 3.14159265358979323846; 	

/**
 * @brief Converts degrees to radians.
 *
 * This function converts an angle from degrees to radians.
 *
 * @param[in] degrees The angle in degrees.
 * @return The angle in radians.
 */
double degrees2rad(double degrees)
{
	return (degrees / 180) * PI;
}

/**
 * @brief Converts radians to degrees.
 *
 * This function converts an angle from radians to degrees.
 *
 * @param[in] radians The angle in radians.
 * @return The angle in degrees.
 */
double rad2degrees(double radians)
{
	return (radians / PI) * 180;
}

/**
 * @brief Comparator function to sort contours by descending area.
 *
 * This function compares the areas of two contours and returns `true` if the area of the first contour is greater than the area of the second contour.
 *
 * @param[in] first The first contour.
 * @param[in] second The second contour.
 * @return `true` if the area of the first contour is greater than the area of the second contour, `false` otherwise.
 *
 * @see cv::contourArea
 */
bool sortByDescendingArea(const object& first, const object& second)
{
	return cv::contourArea(first) > contourArea(second);
}

/**
 * @brief Rotates an image by multiples of 90 degrees clockwise.
 *
 * This function rotates the input image by a specified number of 90-degree steps clockwise.
 * The valid step values are 0 (no rotation), 1 (90 degrees), 2 (180 degrees), and 3 (270 degrees).
 *
 * @param[in] img The input image to be rotated.
 * @param[in] step The number of 90-degree steps to rotate the image. A negative value is converted to its positive equivalent.
 * @return A `cv::Mat` object representing the rotated image.
 *
 * @note The function uses `cv::transpose` and `cv::flip` to perform the rotations.
 * @note If `step` is greater than 3, it is reduced modulo 4 to ensure a valid number of steps.
 *
 * @see cv::transpose
 * @see cv::flip
 */
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
 * @brief Creates a new directory if it does not already exist.
 *
 * This function takes a base folder path and a directory name, creates the directory if it does not already exist,
 * and returns the path to the created directory.
 *
 * @param[in] folder_path The base path where the new directory will be created.
 * @param[in] directory_name The name of the directory to be created.
 * @return A `std::filesystem::path` object representing the path to the created directory.
 *
 * @note If the directory already exists, the function simply returns the path without creating a new directory.
 *
 * @see std::filesystem::exists
 * @see std::filesystem::create_directory
 */
std::filesystem::path createDirectory(const std::filesystem::path& folder_path, const std::string& directory_name)
{
	std::filesystem::path new_dir_path = folder_path / directory_name;

	if (!std::filesystem::exists(new_dir_path))
		std::filesystem::create_directory(new_dir_path);

	return new_dir_path;
}

/**
 * @brief Retrieves a list of file paths that match a specified pattern in a directory.
 *
 * This function uses the OpenCV `cv::glob` function to find all files in the specified directory
 * that match the given pattern and stores the paths of these files in a vector.
 *
 * @param[in] directory The directory in which to search for files.
 * @param[in] pattern The pattern to match files against, such as "*.png".
 * @param[out] file_paths A vector of strings where the matching file paths will be stored.
 *
 * @see cv::glob
 */
void globFiles(const std::string& directory, const std::string& pattern, std::vector<std::string>& file_paths)
{
	const std::string full_pattern = directory + "/" + pattern;
	cv::glob(full_pattern, file_paths);
}


/**
 * @brief Reads images from a list of file paths and stores them in a vector.
 *
 * This function iterates through a list of image file paths, reads each image using OpenCV's `cv::imread` function
 * with the specified flags, and stores the successfully read images in a vector.
 *
 * @param[in] img_paths A vector of strings representing the paths to the image files to be read.
 * @param[out] images A vector of `cv::Mat` objects where the successfully read images will be stored.
 * @param[in] flags The flag that specifies the way the image should be read. This is passed to `cv::imread`.
 *
 * @see cv::imread
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


/**
 * @brief Processes YOLO labels from a file and converts them to bounding boxes.
 *
 * This function reads a YOLO label file and converts the normalized bounding box coordinates to `cv::Rect`
 * objects representing the bounding boxes in the image. The bounding boxes are then stored in a vector.
 *
 * @param[in] filePath The path to the YOLO label file.
 * @param[in] img The image in which the bounding boxes are defined.
 * @param[out] yolo_boxes A vector of `cv::Rect` objects where the converted bounding boxes will be stored.
 *
 * @note The YOLO label file should have the following format for each line:
 *       <class_id> <center_x> <center_y> <width> <height>
 *       where `center_x`, `center_y`, `width`, and `height` are normalized coordinates.
 * @note The function skips lines that cannot be read or have bounding boxes that fall beyond the image boundaries.
 *
 * @see Yolo2BRect
 */
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

/**
 * @brief Converts YOLO format bounding box coordinates to a `cv::Rect`.
 *
 * This function converts normalized YOLO bounding box coordinates (center x, center y, width, height)
 * to a `cv::Rect` with pixel coordinates, ensuring the bounding box is within the image boundaries.
 *
 * @param[in] img The image for which the bounding box is defined.
 * @param[in] x_center The normalized x coordinate of the bounding box center.
 * @param[in] y_center The normalized y coordinate of the bounding box center.
 * @param[in] width The normalized width of the bounding box.
 * @param[in] height The normalized height of the bounding box.
 * @return A `cv::Rect` representing the bounding box in pixel coordinates. If the bounding box falls outside the image boundaries, an empty `cv::Rect` is returned.
 *
 * @note Normalized coordinates are in the range [0, 1]. The function converts these to pixel values.
 * @note If the calculated bounding box exceeds the image boundaries, an empty `cv::Rect` is returned.
 */
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
 * @brief Checks if a region of interest (ROI) is completely within the image boundaries.
 *
 * This function checks whether a given ROI is entirely contained within the boundaries of an image
 * with the specified width and height.
 *
 * @param[in] roi The region of interest represented as a `cv::Rect`.
 * @param[in] width The width of the image.
 * @param[in] height The height of the image.
 * @return `true` if the ROI is completely within the image boundaries, `false` otherwise.
 */
bool isRoiInImage(const cv::Rect& roi,int width, int height)
{
	// Create a rectangle that represents the image boundaries
	const cv::Rect image_rect(0, 0, width, height);

	// Check if the roi is completely inside the image
	return (image_rect & roi) == roi;
}


/**
 * @brief Reads YOLO bounding boxes from a file and converts them to OpenCV `cv::Rect` format.
 *
 * This function reads a file containing YOLO format bounding box coordinates and converts them to
 * OpenCV `cv::Rect` objects, which are stored in a vector. The YOLO format assumes normalized coordinates
 * in the range [0, 1].
 *
 * @param[in] file_path The path to the file containing YOLO bounding box coordinates.
 * @param[in] img The image for which the bounding boxes are defined. Used to convert normalized coordinates to pixel values.
 * @return A vector of `cv::Rect` objects representing the bounding boxes.
 *
 * @throws std::runtime_error If the file cannot be opened.
 *
 * @note The expected format of each line in the file is: "class_id x_center y_center width height".
 * @note If a line cannot be parsed, an error message is printed and the line is skipped.
 */
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



/**
 * @brief Opens a file for writing and returns the output file stream.
 *
 * This function opens a file with the specified filename for writing. If the file cannot be opened,
 * it throws a `std::runtime_error`.
 *
 * @param[in] filename The name of the file to be opened.
 * @return A `std::ofstream` object representing the opened file stream.
 *
 * @throws std::runtime_error If the file cannot be opened.
 */
std::ofstream openFile(const std::string& filename)
{
	std::ofstream file(filename);
	if (!file.is_open())
		throw std::runtime_error("Unable to open file: " + filename);

	return file;
}


/**
 * @brief Generates regions of interest (ROIs) from a set of points and ROI sizes.
 *
 * This function takes a vector of points and a vector of ROI sizes, and generates ROIs centered at each point
 * with each of the given sizes. It ensures that the generated ROIs are within the image boundaries before adding them
 * to the output vector.
 *
 * @param[in] points A vector of `cv::Point` objects representing the centers of the ROIs.
 * @param[in] roi_sizes A vector of `cv::Size` objects representing the sizes of the ROIs.
 * @return A vector of `cv::Rect` objects representing the valid ROIs.
 *
 * @note The function checks if each generated ROI is within the image boundaries before adding it to the output vector.
 * @note The function assumes that the `isRoiInImage` function is defined and checks if an ROI is within the image boundaries.
 *
 * @see isRoiInImage
 */
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
 * @brief Calculates the average dimensions of all images in a specified directory.
 *
 * This function reads all PNG images in the specified directory and calculates the average width and height
 * of the images. It returns the average dimensions as a `cv::Size` object.
 *
 * @param[in] directory_path The path to the directory containing the images.
 * @return A `cv::Size` object representing the average width and height of the images.
 *
 * @throws std::runtime_error If no images are found in the directory or if any image fails to load.
 *
 * @note The function assumes that the images are in PNG format.
 *
 * @see cv::glob
 * @see cv::imread
 * @see cv::Size
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
 * @brief Resizes a vector of images to the same dimensions.
 *
 * This function resizes each image in a vector of images to the specified average dimensions.
 *
 * @param[in,out] clustered_imgs_by_intensity A vector of `cv::Mat` objects representing the images to be resized.
 * @param[in] avg_dim The target dimensions to which all images will be resized.
 *
 * @see cv::resize
 * @see cv::Mat
 * @see cv::Size
 */
void reshape2sameDim(std::vector<cv::Mat>& clustered_imgs_by_intensity,const cv::Size& avg_dim)
{
	for (auto& img : clustered_imgs_by_intensity) 
		cv::resize(img, img, avg_dim);
}


/**
 * @brief Calculates the Euclidean distance between two points.
 *
 * This function computes the Euclidean distance between two points `a` and `b` using the OpenCV `cv::norm` function.
 *
 * @param[in] a The first point.
 * @param[in] b The second point.
 * @return The Euclidean distance between the two points.
 *
 * @see cv::Point
 * @see cv::norm
 */
double euclidean_distance(const cv::Point& a, const cv::Point& b)
{
	return cv::norm(a - b);
}

/**
 * @brief Filters a set of points by a minimum distance criterion.
 *
 * This function filters a vector of points, selecting only those points that are not closer than
 * a specified minimum distance to any previously selected points. The points are first sorted
 * by their coordinates to ensure consistent results.
 *
 * @param[in,out] points A vector of `cv::Point` objects representing the points to be filtered.
 *                       The vector is sorted by coordinates.
 * @param[in] min_distance The minimum distance required between any two selected points.
 * @return A vector of `cv::Point` objects representing the points that meet the minimum distance criterion.
 *
 * @see euclidean_distance
 * @see cv::Point
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
 * @brief Recursively lists all leaf directories within a specified directory.
 *
 * This function traverses the directory tree starting from the specified directory path
 * and collects the paths of all leaf directories (directories that do not contain any subdirectories).
 *
 * @param[in] directory_path The path to the directory to start the traversal from.
 * @param[out] final_paths A vector of strings to store the paths of all leaf directories.
 *
 * @note A leaf directory is defined as a directory that does not contain any subdirectories.
 * @note The function uses recursion to traverse the directory tree.
 *
 * @see std::filesystem::directory_iterator
 * @see std::filesystem::path
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