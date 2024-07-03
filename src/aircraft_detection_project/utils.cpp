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


// utility function that rotates 'img' by step*90�
// step = 0 --> no rotation
// step = 1 --> 90� CW rotation
// step = 2 --> 180� CW rotation
// step = 3 --> 270� CW rotation
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
	// 90� CW rotation
	else if (step == 1)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 1);
	}
	// 180� CW rotation
	else if (step == 2)
		cv::flip(img, img_rot, -1);
	// 270� CW rotation
	else if (step == 3)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 0);
	}

	return img_rot;
}


std::filesystem::path createDirectory(const std::filesystem::path& folder_path, const std::string& directory_name)
{
	std::filesystem::path new_dir_path = folder_path / directory_name;

	if (!std::filesystem::exists(new_dir_path))
		std::filesystem::create_directory(new_dir_path);

	return new_dir_path;
}


void globFiles(const std::string& directory, const std::string& pattern, std::vector<std::string>& file_paths)
{
	const std::string full_pattern = directory + "/" + pattern;
	cv::glob(full_pattern, file_paths);
}


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
	// center_x --> the normalized x coord of the bounding box center.
	// center_y --> the normalized y coord of the bounding box center.
	// width    --> the normalized width   of the bounding box.
	// height   --> the normalized height  of the bounding box.
	if (file.is_open())
	{
		std::string line;
		while (std::getline(file, line))
		{
			std::istringstream lineStream(line);
			int class_id;
			double center_x, center_y, width, height;
			if (!(lineStream >> class_id >> center_x >> center_y >> width >> height))
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
	// Convert normalized coordinates to pixel values
	int x_center_px = ucas::round<float>(x_center * img.cols);
	int y_center_px = ucas::round<float>(y_center * img.rows);
	int width_px    = ucas::round<float>(width * img.cols);
	int height_px   = ucas::round<float>(height * img.rows);

	// Calculate top left corner of bounding box
	int x = x_center_px - width_px / 2;
	int y = y_center_px - height_px / 2;

	// Check if the bounding box falls beyond the image boundaries: if so an empty cv::Rect is returned
	if (x < 0 || y < 0 || x + width_px > img.cols || y + height_px > img.rows)
	{
		std::cerr << "Error: Bounding box falls beyond the image boundaries.\n";
		return {};
	}

	return { x, y, width_px, height_px };
}


bool isRoiInImage(const cv::Rect& roi,int width, int height)
{
	// Crea un rettangolo che rappresenta l'intera immagine
	cv::Rect imageRect(0, 0, width, height);

	// Verifica se il ROI � contenuto completamente nel rettangolo dell'immagine
	return (imageRect & roi) == roi;
}



std::vector<cv::Rect> readYoloBoxes(const std::filesystem::path& file_path)
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
			std::cerr << "Error parsing line: " << line << std::endl;
			continue;
		}

		// Convert YOLO format to OpenCV cv::Rect
		int x = static_cast<int>((x_center - width / 2) * 4800);
		int y = static_cast<int>((y_center - height / 2) * 2703);
		int w = static_cast<int>(width * 4800);
		int h = static_cast<int>(height * 2703);

		yolo_boxes.emplace_back(x, y, w, h);
	}

	infile.close();
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

std::vector<cv::Rect> generateRoisFromPoints(const std::vector<cv::Point>& points, const std::array<cv::Size, 5>& roi_sizes)
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


cv::Size calculateAvgDims(const std::filesystem::path& directory_path)
{
	std::vector<std::string> image_paths;
	cv::glob(directory_path.string() + "/*.png", image_paths);

	if (image_paths.empty()) 
		throw std::runtime_error("No images found in the directory.");


	int acc_widths = 0;
	int acc_heights = 0;
	std::vector<cv::Mat> images;

	for (const auto& image_path : image_paths) {
		cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
		if (img.empty()) 
			throw std::runtime_error("Failed to load image: " + image_path);
		
		images.push_back(img);
		acc_widths += img.cols;
		acc_heights += img.rows;
	}

	int avg_width = static_cast<int>(std::round(static_cast<float>(acc_widths) / images.size()));
	int avg_height = static_cast<int>(std::round(static_cast<float>(acc_heights) / images.size()));

	return cv::Size(avg_width, avg_height);
}

void reshape2sameDim(std::vector<cv::Mat>& clustered_imgs_by_intensity,const cv::Size& avg_dim)
{

	for (auto& img : clustered_imgs_by_intensity) 
		cv::resize(img, img, avg_dim);
	
}


/*
void reshape2sameDim(std::vector<cv::Mat>& cluestred_imgs_by_intensity, std::vector<cv::Size>& avg_dims)
{
	int acc_widths = 0;
	int acc_heights = 0;
	for (const auto& img : cluestred_imgs_by_intensity)
	{
		acc_widths += img.cols;
		acc_heights += img.rows;
	}


	int avg_width = ucas::round<float>(acc_widths / cluestred_imgs_by_intensity.size());

	int avg_height = ucas::round<float>(acc_heights / cluestred_imgs_by_intensity.size());

	avg_dims.push_back(cv::Size(avg_width, avg_height));

	// Resizing each image so that it reaches the maximum size
	for (auto& img : cluestred_imgs_by_intensity)
		cv::resize(img, img, cv::Size(avg_width, avg_height), 1, 1, cv::INTER_CUBIC);
}
*/


double euclidean_distance(const cv::Point& a, const cv::Point& b)
{
	return cv::norm(a - b);
}

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