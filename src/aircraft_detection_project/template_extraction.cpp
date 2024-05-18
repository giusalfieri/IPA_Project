#include "template_extraction.h"


typedef std::vector <cv::Point>  object;



double degrees2rad(double degrees)
{
	return (degrees / 180) * ipa::PI;
}

double rad2degrees(double radians)
{
	return (radians / ipa::PI) * 180;
}


// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
cv::Mat rotate90(cv::Mat img, int step);

cv::Rect Yolo2BRect(const cv::Mat& input_img, double x_center, double y_center, double width, double height);

bool sortByDescendingArea(object& first, object& second);

cv::Mat getRotationROI(cv::Mat& img, cv::Rect& roi);


void ExtractTemplates()
{

	// Utilizza glob per ottenere i nomi dei file corrispondenti al pattern
	std::vector<std::string> img_paths;
	std::string pattern1 = DATASET_PATH + std::string("/*.jpg");
	cv::glob(pattern1, img_paths);
	std::sort(img_paths.begin(), img_paths.end());

	std::vector<std::string> yolo_paths;
	std::string pattern2 = DATASET_PATH + std::string("/*.txt");
	cv::glob(pattern2, yolo_paths);
	std::sort(yolo_paths.begin(), yolo_paths.end());

	/*
	for (const auto& names : img_paths)
	{
		std::cout << names << "\n";
	}

	for(const auto& names: yolo_paths)
	{
		std::cout << names << "\n";
	}
	*/

	for (int i = 0; i < img_paths.size(); i++)
	{


		std::cout << "\n---------------------------------------\n";
		std::cout << "Starting processing image " << i;
		std::cout << "\n---------------------------------------\n\n";

		cv::Mat img = cv::imread(img_paths[i]);



		cv::Mat clone = img.clone();
		cv::Mat img_HSV = img.clone();
		cv::cvtColor(img_HSV, img_HSV, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> channels;
		cv::split(img_HSV, channels);



		std::ifstream file(yolo_paths[i]);


		std::vector <cv::Rect> airplanes_boxes;


		std::vector<cv::Mat> final_templates;


		if (file.is_open()) {

			std::string line;

			while (std::getline(file, line))
			{

				std::istringstream iss(line);
				int class_id;
				double center_x, center_y, width, height;
				if (!(iss >> class_id >> center_x >> center_y >> width >> height))
					break; // error in reading

				airplanes_boxes.push_back(Yolo2BRect(img, center_x, center_y, width, height));
			}

			file.close();
		}

		std::vector<cv::Mat> airplanes_to_be_processed;
		std::vector<cv::Rect> yolo_boxes_to_be_processed;
		for (const auto& box : airplanes_boxes)
		{
			cv::Mat airplane = img(box).clone(); // review carefully if clone is needed


			std::string answer;
			std::cout << "Process this airplane?\n";
			cv::imshow("Template 12", airplane);
			cv::waitKey(100);
			do
			{
				std::cin >> answer;
				if (answer != "Y" && answer != "y" && answer != "N" && answer != "n")
					std::cout << "Invalid input! Please enter:  Y/y(yes) N/n(no)\n";

			} while (answer != "Y" && answer != "y" && answer != "N" && answer != "n");

			if (answer == "Y" || answer == "y")
				yolo_boxes_to_be_processed.push_back(box);
			else
				final_templates.push_back(airplane);
		}

		std::vector<cv::Mat> binarized_airplanes;
		for (const auto& box : yolo_boxes_to_be_processed)
		{
			cv::Mat airplane = channels[2](box).clone(); // review carefully if clone is needed


			cv::Mat img_bin_adaptive;
			int block_size = airplane.cols;
			if (block_size % 2 == 0)
			{
				block_size += 1;
			}
			int C = -20;
			cv::adaptiveThreshold(airplane, img_bin_adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, C);
			cv::morphologyEx(img_bin_adaptive, img_bin_adaptive, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
			//ipa::imshow("Adaptive Binarized", img_bin_adaptive, true);
			binarized_airplanes.push_back(img_bin_adaptive);

		}


		std::vector<std::pair<cv::Point2f, double>> geometric_moments_descriptors;

		for (int i = 0; i < binarized_airplanes.size(); i++)
		{
			std::vector< std::vector <cv::Point> > contours;

			cv::findContours(binarized_airplanes[i], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

			std::sort(contours.begin(), contours.end(), sortByDescendingArea);


			cv::Moments moments = cv::moments(contours[0], true);
			cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);

			cv::Point2f corrected_center = cv::Point2f(center.x + airplanes_boxes[i].x,
				center.y + airplanes_boxes[i].y);


			double angle = 0.5 * std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

			geometric_moments_descriptors.push_back(std::make_pair(corrected_center, rad2degrees(angle)));

		}


		std::vector<double> corrected_angle;
		std::vector<cv::Mat> templates_vector;
		for (int i = 0; i < binarized_airplanes.size(); i++)
		{

			cv::Point2f center = geometric_moments_descriptors[i].first;
			double angle = geometric_moments_descriptors[i].second;

			// Angle correction
			if (angle >= 0 && angle <= 90)
				angle += 90.0f;
			else if (angle < 0 && angle >= -90)
				angle = 90.0f - (-angle);

			corrected_angle.push_back(angle);


			int h = 2;
			cv::Rect roi(center.x - airplanes_boxes[i].width * h / 2, center.y - airplanes_boxes[i].height * h / 2, airplanes_boxes[i].width * h, airplanes_boxes[i].height * h);
			cv::Mat rotation_roi = getRotationROI(img, roi);;


			cv::Point roi_center = cv::Point(ucas::round<float>(rotation_roi.cols / 2), ucas::round<float>(rotation_roi.rows / 2));
			cv::Mat rotation_mat = cv::getRotationMatrix2D(roi_center, angle, 1);

			cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), rotation_roi.size(), angle).boundingRect2f();
			// adjust transformation matrix
			rotation_mat.at<double>(0, 2) += bbox.width / 2.0 - rotation_roi.cols / 2.0;
			rotation_mat.at<double>(1, 2) += bbox.height / 2.0 - rotation_roi.rows / 2.0;

			cv::Mat dst;
			cv::warpAffine(rotation_roi, dst, rotation_mat, bbox.size());


			float scaling_factor = 1.15;
			cv::Rect final_roi(dst.cols / 2 - airplanes_boxes[i].width * scaling_factor / 2, dst.rows / 2 - airplanes_boxes[i].height * scaling_factor / 2, airplanes_boxes[i].width * scaling_factor, airplanes_boxes[i].height * scaling_factor);

			cv::Mat final_template = dst(final_roi);

			templates_vector.push_back(final_template);
		}


		std::string answer;
		for (auto& airplane_template : templates_vector)
		{

			std::cout << "Do you wish to keep this template? Y/y(yes) N/n(no): ";
			cv::imshow("Template", airplane_template);
			cv::waitKey(100);
			//cv::destroyWindow("Template");
			cv::Mat airplane = airplane_template.clone();

			do
			{
				std::cin >> answer;
				if (answer != "Y" && answer != "y" && answer != "N" && answer != "n")
					std::cout << "Invalid input! Please enter:  Y/y(yes) N/n(no)\n";

			} while (answer != "Y" && answer != "y" && answer != "N" && answer != "n");


			if (answer == "Y" || answer == "y")
			{
				std::cout << "Perform rotation?  Y/y(yes) N/n(no)\n";
				do
				{
					std::cin >> answer;
					if (answer != "Y" && answer != "y" && answer != "N" && answer != "n")
						std::cout << "Invalid input! Please enter:  Y/y(yes) N/n(no)\n";

				} while (answer != "Y" && answer != "y" && answer != "N" && answer != "n");

			}
			else continue;

			if (answer == "Y" || answer == "y")
			{
				std::cout << "Please provide a rotation type (0-->no rotation, 1-->CW rotation 90, 2-->CW rotation 180, 3-->CW rotation 270): ";
				do
				{
					std::cin >> answer;
					if (answer != "0" && answer != "1" && answer != "2" && answer != "3")
						std::cout << "Invalid input! Please enter a type (0-->no rotation, 1-->CW rotation 90, 2-->CW rotation 180, 3-->CW rotation 270): \n";

				} while (answer != "0" && answer != "1" && answer != "2" && answer != "3");

				airplane = rotate90(airplane_template, std::stoi(answer));

			}



			final_templates.push_back(airplane);
		}





		for (const auto& airplane_template : final_templates)
		{
			// std::cout << airplane_template.cols << " " << airplane_template.rows << "\n";
			//ipa::imshow("Template", airplane_template);
		  	cv::imwrite("sa", airplane_template);
			//cv::waitKey(3000);
		}

		std::cout << "\n---------------------------------------\n";
		std::cout << "Image " << i << " has been processed successfully";
		std::cout << "\n---------------------------------------\n\n";

	}

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

cv::Rect Yolo2BRect(const cv::Mat& input_img, double x_center, double y_center, double width, double height)
{
	int imageWidth = input_img.cols;
	int imageHeight = input_img.rows;

	// Convert normalized coordinates to pixel values
	int x_center_px = ucas::round<float>(x_center * imageWidth);
	int y_center_px = ucas::round<float>(y_center * imageHeight);
	int width_px = ucas::round<float>(width * imageWidth);
	int height_px = ucas::round<float>(height * imageHeight);

	// Calculate top left corner of bounding box
	int x = x_center_px - width_px / 2;
	int y = y_center_px - height_px / 2;

	return cv::Rect(x, y, width_px, height_px);
}

bool sortByDescendingArea(object& first, object& second)
{
	return cv::contourArea(first) > contourArea(second);
}


cv::Mat getRotationROI(cv::Mat& img, cv::Rect& roi) {
	cv::Mat rotation_roi;

	// Check if the ROI is within the image boundaries
	if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= img.cols && roi.y + roi.height <= img.rows)
	{
		rotation_roi = img(roi);
	}
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
