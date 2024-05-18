
#include "template_extraction.h"
#include "utils.h"



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

	// Ottieni il percorso della directory padre
	std::filesystem::path dataset_path(DATASET_PATH);
	std::filesystem::path parent_path = dataset_path.parent_path();



	// Crea un nuovo percorso per la nuova directory
	std::filesystem::path new_dir_path = parent_path / "extracted_templates";

	try 
	{
		// Prova a creare la nuova directory solo se non esiste già
		if (!std::filesystem::exists(new_dir_path)) 
			std::filesystem::create_directory(new_dir_path);
	}
	catch (std::filesystem::filesystem_error& e) 
	{
		// Gestisci l'eccezione
		std::cout << "Errore durante la creazione della directory: " << e.what() << std::endl;
	}



	std::vector<cv::Mat> final_templates;


	for (int i = 0; i < img_paths.size(); i++)
	{

		//Aggiungere qui il proprio dataset

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



		


		std::cout << "\n---------------------------------------\n";
		std::cout << "Image " << i << " has been processed successfully";
		std::cout << "\n---------------------------------------\n\n";

	}


	const std::string name = "giuseppe";
	
	for (int k = 0; k < final_templates.size(); k++)
	{
		const std::string str = std::to_string(k);
		const std::string template_name = "template" + str + "_"+ name + ".png";
		// Crea il percorso del file di output
		std::filesystem::path output_path = parent_path / template_name;
		// Salva l'immagine
		cv::imwrite(output_path.string(), final_templates[k]);
	}

	

}



// 	void KMeansClusteringSizes()
// 	{


// 		std::filesystem::path template_path();
		


// 		std::vector<cv::Mat> images;



// 		cv::Mat sizes(images.size(), 2, CV_32F);
	
// 		for (int i = 0; i < images.size(); i++)
// 		{
// 			sizes.at<float>(i, 0) = images[i].cols;
// 			sizes.at<float>(i, 1) = images[i].rows;
// 		}
		
// 		cv::Mat labels, centers;
		
// 		cv::kmeans(
// 		sizes, 
// 		k, labels, cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 
// 		3, 
// 		cv::KMEANS_RANDOM_CENTERS,
// 		centers
// 		);
		
// 		for (int i = 0; i < k; i++)
// 		{
// 			std::cout << "Cluster " << i << ":\n";
// 			for (int j = 0; j < labels.rows; j++)
// 			{
// 				if (labels.at<int>(j) == i)
// 				{
// 					std::cout << "Image " << j << ": " << images[j].cols << "x" << images[j].rows << "\n";
// 				}
// 			}
// 			std::cout << "\n";
// 		}
// 	}


// void KMeansClusteringIntensities(){
	



// }


