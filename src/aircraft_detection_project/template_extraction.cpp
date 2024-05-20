#include "template_extraction.h"
#include "utils.h"



std::string getValidInput(const std::string& prompt, const std::vector<std::string>& valid_inputs);

cv::Mat getRotationROI(cv::Mat& img, cv::Rect& roi);

cv::Rect Yolo2BRect(const cv::Mat& img, double x_center, double y_center, double width, double height);



void extractTemplates()
{

	std::vector<std::string> dataset_img_paths;
	auto dataset_img_paths_pattern = DATASET_PATH + std::string("/*.jpg");
	cv::glob(dataset_img_paths_pattern, dataset_img_paths);

	std::vector<std::string> yolo_labels_paths;
	auto yolo_labels_paths_pattern = DATASET_PATH + std::string("/*.txt");
	cv::glob(yolo_labels_paths_pattern, yolo_labels_paths);


	// Ottieni il percorso della directory padre
	auto extracted_templates_path = std::filesystem::path(DATASET_PATH).parent_path();

	auto extracted_templates_folder = createDirectory(extracted_templates_path, "extracted_templates");

	int count = 0;

	for (int k = 0; k < dataset_img_paths.size(); k++)
	{

		std::cout << "\n---------------------------------------\n";
		std::cout << "      Starting processing image " << k;
		std::cout << "\n---------------------------------------\n\n";

		cv::Mat img = cv::imread(dataset_img_paths[k]);
		std::filesystem::path p(dataset_img_paths[k]);
		auto img_filename = p.stem().string();

		cv::Mat clone = img.clone();
		cv::Mat img_HSV = img.clone();
		cv::cvtColor(img_HSV, img_HSV, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> channels;
		cv::split(img_HSV, channels);


		std::ifstream file(yolo_labels_paths[k]);


		std::vector <cv::Rect> yolo_boxes;

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



		std::vector<cv::Rect> selected_airplanes_yolo_boxes;
		for (const auto& box : yolo_boxes)
		{
			cv::Mat airplane = img(box).clone(); // review carefully if clone is needed
			std::cout << "ciao" << "\n";
			cv::imshow("Airplane", airplane);
			cv::waitKey(100);
			std::string answer = getValidInput("Process this airplane? Y/y(yes) N/n(no): ", { "Y", "y", "N", "n" });

			if (answer == "Y" || answer == "y")
				selected_airplanes_yolo_boxes.push_back(box);
			else
			{
				const auto template_name = "template_" + std::to_string(count) + "_" + img_filename;
				std::filesystem::path output_path = extracted_templates_folder / (template_name + ".png");
				cv::imwrite(output_path.string(), airplane);
				count++;
				
			}

		}

		// Binarization step 
		std::vector<cv::Mat> bin_airplanes;
		for (const auto& box : selected_airplanes_yolo_boxes)
		{
			
			cv::Mat airplane = channels[2](box).clone(); // review carefully if clone is needed

			cv::Mat img_bin_adaptive;
			int block_size = airplane.cols;
			if (block_size % 2 == 0)
				block_size += 1;
			int C = -20;
			cv::adaptiveThreshold(airplane, img_bin_adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, C);
			cv::morphologyEx(img_bin_adaptive, img_bin_adaptive, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
			//ipa::imshow("Adaptive Binarized", img_bin_adaptive, true);
			bin_airplanes.push_back(img_bin_adaptive);
		}


		std::vector<std::pair<cv::Point2f, double>> geometric_moments_descriptors;
		for (int i = 0; i < bin_airplanes.size(); i++)
		{
			std::vector< std::vector<cv::Point> > contours;

			cv::findContours(bin_airplanes[i], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

			std::sort(contours.begin(), contours.end(), sortByDescendingArea);

			cv::Moments moments = cv::moments(contours[0], true);

			cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);

			double angle = 0.5 * std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

			// Centroid correction
			cv::Point2f corrected_center = cv::Point2f(center.x + yolo_boxes[i].x,
				center.y + yolo_boxes[i].y);

			// Angle correction
			if (angle >= 0 && angle <= 90)
				angle += 90.0f;
			else if (angle < 0 && angle >= -90)
				angle = 90.0f - (-angle);

			geometric_moments_descriptors.push_back(std::make_pair(corrected_center, rad2degrees(angle)));
		}


		std::vector<cv::Mat> templates_vector;
		for (int i = 0; i < bin_airplanes.size(); i++)
		{
			cv::Point2f center = geometric_moments_descriptors[i].first;
			double angle = geometric_moments_descriptors[i].second;

			float h = 2.0f;
			cv::Rect roi(ucas::round<float>(center.x - (yolo_boxes[i].width * h) / 2.0f),
						 ucas::round<float>(center.y - (yolo_boxes[i].height * h) / 2.0f),
				         ucas::round<float>(yolo_boxes[i].width * h), ucas::round<float>(yolo_boxes[i].height * h));
			cv::Mat rotation_roi = getRotationROI(img, roi);


			cv::Point roi_center = cv::Point(ucas::round<float>(rotation_roi.cols / 2.0f), ucas::round<float>(rotation_roi.rows / 2.0f));
			cv::Mat rotation_mat = cv::getRotationMatrix2D(roi_center, angle, 1);

			cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), rotation_roi.size(), angle).boundingRect2f();

			// adjust transformation matrix
			rotation_mat.at<double>(0, 2) += bbox.width / 2.0f - rotation_roi.cols / 2.0f;
			rotation_mat.at<double>(1, 2) += bbox.height / 2.0f - rotation_roi.rows / 2.0f;

			cv::Mat dst;
			cv::warpAffine(rotation_roi, dst, rotation_mat, bbox.size());


			float scaling_factor = 1.15;

			cv::Rect final_roi(ucas::round<float>(dst.cols / 2.0f - (yolo_boxes[i].width * scaling_factor) / 2.0f),
				ucas::round<float>(dst.rows / 2.0f - (yolo_boxes[i].height * scaling_factor) / 2.0f),
				ucas::round<float>(yolo_boxes[i].width * scaling_factor),
				ucas::round<float>(yolo_boxes[i].height * scaling_factor));

			
			cv::Mat final_template = dst(final_roi);
		

			templates_vector.push_back(final_template);
		}



		for (auto& airplane_template : templates_vector)
		{
			cv::imshow("Template", airplane_template);
			cv::waitKey(100);
			cv::Mat airplane = airplane_template.clone();

			auto answer = getValidInput("Do you wish to keep this template? Y/y(yes) N/n(no): ", { "Y", "y", "N", "n" });
			if (answer == "Y" || answer == "y")
			{
				answer = getValidInput("Perform rotation?  Y/y(yes) N/n(no)\n", { "Y", "y", "N", "n" });
				if (answer == "Y" || answer == "y")
				{
					answer = getValidInput("Please provide a rotation type (0-->no rotation, 1-->CW rotation 90, 2-->CW rotation 180, 3-->CW rotation 270): ", { "0", "1", "2", "3" });
					airplane = rotate90(airplane_template, std::stoi(answer));
				}
			}
			else continue;



			const auto template_name = "template" + std::to_string(count) + "_" + img_filename;
			std::filesystem::path output_path = extracted_templates_folder / (template_name + ".png");
			cv::imwrite(output_path.string(), airplane);
			count++;

		}


		std::cout << "\n---------------------------------------\n";
		std::cout << "Image " << k << " has been processed successfully";
		std::cout << "\n---------------------------------------\n\n";
	}


}

std::string getValidInput(const std::string& prompt, const std::vector<std::string>& valid_inputs)
{
	std::string answer;
	do
	{
		std::cout << prompt;
		std::cin >> answer;
		if (std::find(valid_inputs.begin(), valid_inputs.end(), answer) == valid_inputs.end())
		{
			std::cout << "Invalid input! Please enter one of the following: ";
			for (const auto& input : valid_inputs)
				std::cout << input << " ";
			std::cout << "\n";
		}
	} while (std::find(valid_inputs.begin(), valid_inputs.end(), answer) == valid_inputs.end());
	return answer;
}


cv::Mat getRotationROI(cv::Mat& img, cv::Rect& roi)
{


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
	if (x < 0 || y < 0 || x + width_px > img.cols || y + height_px > img.rows) {
		std::cerr << "Error: Bounding box falls beyond the image boundaries.\n";
		return cv::Rect();
	}

	return cv::Rect(x, y, width_px, height_px);
}
