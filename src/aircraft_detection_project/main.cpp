#include "template_extraction.h"
#include "kmeans.h"
#include "utils.h"
#include "eigenplanes.h"


bool pointInBoxes(const cv::Point2f& point, const std::vector<cv::RotatedRect>& boxes);

int main()
{
	/*/*
    ------------------------------------------
	**** TEMPLATE EXTRACTION FROM DATASET ****
	------------------------------------------
	#1#
    //extractTemplates();
	//----------------------------------------





	/*
	------------------------------------------
	*********  K-MEANS  CLUSTERING  **********  
	------------------------------------------
	#1#

	/*
	--------------------------
	1)  K-MEANS BY IMAGES SIZE     
    --------------------------
	#1#

	//--------------------
	// 1.1) Reading images 
	//--------------------
	const auto extracted_templates_folder_path = std::filesystem::path(DATASET_PATH).parent_path() / "extracted_templates";

	std::vector<std::string> template_paths;
	globFiles(extracted_templates_folder_path.string(), "/*.png", template_paths);
	std::vector<cv::Mat> extracted_templates;
	readImages(template_paths, extracted_templates);
	/*
	std::vector<std::string> template_paths;
	globFiles(extracted_templates_folder_path.string(), "/*.png", template_paths);
	
	std::vector<cv::Mat> extracted_templates;
	for (const auto& path : template_paths)
	{
		cv::Mat img = cv::imread(path);
		if (img.data)
			extracted_templates.push_back(img);
	}
	#1#
	//--------------------


	//------------------------
	// 1.2) Performing K-Means
	//------------------------
	const int num_clusters_by_size =  5;
	cv::Mat labels = kmeansBySize(extracted_templates, num_clusters_by_size);
	//------------------------


	//----------------------------------------------
	// 1.3) Creating folders to store image clusters
	//----------------------------------------------
	std::filesystem::path kmean_by_size_folder_path = createDirectory(std::filesystem::path(DATASET_PATH).parent_path(), "kmeans_by_size");
	std::filesystem::path cluster_by_size_folder_path;
	std::vector<std::filesystem::path> clusters_by_size_paths;
	for (int i = 0; i < num_clusters_by_size; i++)
	{
		cluster_by_size_folder_path = createDirectory(kmean_by_size_folder_path, "Cluster_" + std::to_string(i));
		clusters_by_size_paths.push_back(cluster_by_size_folder_path);
	}
	//----------------------------------------------


	//-----------------------------
	// 1.4) Saving clustered images 
	//-----------------------------
	for (int j = 0; j < extracted_templates.size(); j++)
	{
		// Get the cluster label for the current template
		int cluster_id = labels.at<int>(j);
		const auto extracted_template_id = std::filesystem::path(template_paths[j]).stem();
		const std::filesystem::path clusterd_template_path = clusters_by_size_paths[cluster_id] / extracted_template_id;

		// Save the template image to the corresponding cluster folder
		cv::imwrite(clusterd_template_path.string() + ".png", extracted_templates[j]);
	}
	//----------------------------
	



	/*
	---------------------------------
	2)  K-MEANS BY IMAGES INTENSITIES
	---------------------------------
	#1#

	const auto kmeans_intensity_folder_path = createDirectory(std::filesystem::path(DATASET_PATH).parent_path(), "kmeans_by_intensity");
	std::vector<std::filesystem::path> final_clusters_by_intensity_paths;
	const int num_clusters_by_intensity = 5;
	for (int k = 0; k < num_clusters_by_size; k++)
	{

		std::vector<std::string> clustered_by_size_templates_path;
		globFiles(clusters_by_size_paths[k].string(), "/*.png", clustered_by_size_templates_path);
	
		std::vector<cv::Mat> clustered_by_size_templates;
		//readImages(clustered_by_size_templates_path, clustered_by_size_templates);
		
		for (const auto& path : clustered_by_size_templates_path)
		{
			cv::Mat template_gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
			if (template_gray.data)
				clustered_by_size_templates.push_back(template_gray);
		}
		




		//------* Performing K-Means ClusteringBy Size *---      
		cv::Mat labels_intensity_clusters = kmeansByIntensity(clustered_by_size_templates, num_clusters_by_intensity);
		//-------------------------------------------------




		const auto kmeans_intensity_cluster_group_path = createDirectory(kmeans_intensity_folder_path, "Group_" + std::to_string(k));

		std::vector<std::filesystem::path> clusters_by_intensity_paths;
		for (int j = 0; j < num_clusters_by_intensity; j++)
		{
			const auto ith_cluster_by_intensity_folder_path = createDirectory(kmeans_intensity_cluster_group_path, "Cluster_By_Intensity_" + std::to_string(j));
			clusters_by_intensity_paths.push_back(ith_cluster_by_intensity_folder_path);
			final_clusters_by_intensity_paths.push_back(ith_cluster_by_intensity_folder_path);
		}
		

		



		//--------*  Saving clustered images *-------------
		for (int j = 0; j < clustered_by_size_templates.size(); j++)
		{
			// Get the cluster label for the current template
			int cluster_by_intensity_id = labels_intensity_clusters.at<int>(j);
			const auto cluestered_by_intensity_template_id = std::filesystem::path(clustered_by_size_templates_path[j]).stem();
			const std::filesystem::path clusterd_by_intensity_template_path = clusters_by_intensity_paths[cluster_by_intensity_id] / cluestered_by_intensity_template_id;

			// Save the template image to the corresponding cluster folder
			cv::imwrite(clusterd_by_intensity_template_path.string() + ".png", clustered_by_size_templates[j]);
		}
		//************************************************ 
	}
	


	/*
	-------------------------------------------------------------
	  RESIZING THE IMAGES IN EACH CLLUSTER TO BE OF THE SAME SIZE
	-------------------------------------------------------------
	#1#

	/*
	* std::vector<cv::Size> avg_dims;
	* for (int i = 0; i < final_clusters_num; i++)
	{
		std::vector<std::string>final_clusters_paths;
		globFiles(final_clusters_by_intensity_paths[i].string(), "/*.png", final_clusters_paths);

		std::vector<cv::Mat> intensities_img;



		for (const auto& path : final_clusters_paths)
		{
			cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
			if (img.data)
				intensities_img.push_back(img);
		}

		int acc_widths = 0;
		int acc_heights = 0;
		for(const auto& img: intensities_img)
		{
			acc_widths += img.cols;
			acc_heights += img.rows;
		}


		int avg_width = ucas::round<float>(acc_widths / intensities_img.size());

		int avg_height = ucas::round<float>(acc_heights / intensities_img.size());

		avg_dims.push_back(cv::Size(avg_width,avg_height))

	}
	#1#
	std::vector<cv::Size> avg_dims;
	std::vector<int> img_heights_in_final_clusters;
	std::vector<std::filesystem::path> same_size_clusters;
	const auto final_directory_sam_size = createDirectory(std::filesystem::path(DATASET_PATH).parent_path(), "clusters_same_size_imgs");

	const int final_clusters_num = num_clusters_by_size * num_clusters_by_intensity;
	for (int i = 0; i < final_clusters_num; i++)
	{
		std::vector<std::string>final_clusters_paths;
		globFiles(final_clusters_by_intensity_paths[i].string(), "/*.png", final_clusters_paths);

		std::vector<cv::Mat> intensities_img;


		
		for (const auto& path : final_clusters_paths)
		{
			cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
			if (img.data)
				intensities_img.push_back(img);

		}

		const auto ith_cluster_by_intensity_folder_path = createDirectory(std::filesystem::path(final_directory_sam_size), "Cluster_same_size_" + std::to_string(i));
		same_size_clusters.push_back(ith_cluster_by_intensity_folder_path);

		reshape2sameDim_sper(intensities_img,avg_dims);
		//reshape2sameDim(intensities_img);


		bool actionDone = false;
		for (const auto& img : intensities_img)
		{
			if (!actionDone)
			{
				img_heights_in_final_clusters.push_back(img.rows);
				actionDone = true;
			}
		}


		for(int j=0; j < intensities_img.size(); j++)
		{
			const auto  final_template_id = std::filesystem::path(final_clusters_paths[j]).stem();
			const std::filesystem::path clusterd_by_intensity_template_path = ith_cluster_by_intensity_folder_path / final_template_id;

			// Save the template image to the corresponding cluster folder
			cv::imwrite(clusterd_by_intensity_template_path.string() + ".png", intensities_img[j]);
		}

	}


	for (const auto& elem : img_heights_in_final_clusters)
		std::cout << "img_height " << elem << "\n";



	/*
	------------------------------------------------
	  EIGENPLANES (via Principal Component Analysis)
	------------------------------------------------
	#1#
	const auto avg_airplanes_dir = createDirectory(std::filesystem::path(DATASET_PATH).parent_path(), "avg_airplanes_dir");
	for (int i = 0; i < final_clusters_num; i++)
	{

		std::vector<std::string>final_clusters_paths;
		globFiles(same_size_clusters[i].string(), "/*.png", final_clusters_paths);

		std::vector<cv::Mat> intensities_img;
		for (const auto& path : final_clusters_paths)
		{
			cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
			if (img.data)
				intensities_img.push_back(img);
		}
		

		//cv::Mat avg_airplane = eigenPlanes(intensities_img, img_heights_in_final_clusters[i]);
		cv::Mat avg_airplane = eigenPlanes_sper(intensities_img, avg_dims[i]);

		const std::string avg_airplane_name = "avg_airplane" + std::to_string(i);
		// Save the template image to the corresponding cluster folder
		cv::imwrite(avg_airplanes_dir.string() + "/" + avg_airplane_name + ".png", avg_airplane);
	}
	*/
	

	/*
	-------------------
	 TEMPLATE MATCHING
	-------------------
	*/

	std::vector<std::string>avg_airplanes_paths;
	globFiles("C:\\Users\\Giuseppe\\Desktop\\IPA_Project-another_sperimental\\src\\avg_airplanes_dir", "/*.png", avg_airplanes_paths);

	std::vector<cv::Mat> avg_plane;
	for (const auto& path : avg_airplanes_paths)
	{
		cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
		if (img.data)
			avg_plane.push_back(img);
	}
	const auto num_avg_airplanes = avg_plane.size();


	cv::Mat src_img = cv::imread(std::string(DATASET_PATH)+"/GBG_439.jpg",cv::IMREAD_GRAYSCALE);
	for (int i = 0; i < num_avg_airplanes; i++)
	{

	    for( int degree_angle = 0; degree_angle < 360; degree_angle += 5)
	    {
			
			cv::Point rot_center = cv::Point(ucas::round<float>(src_img.cols / 2.0f),
			                  	             ucas::round<float>(src_img.rows / 2.0f));

			cv::Mat rotation_mat = cv::getRotationMatrix2D(rot_center, degree_angle, 1);

			cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src_img.size(), degree_angle).boundingRect2f();

			// adjust transformation matrix
			rotation_mat.at<double>(0, 2) += bbox.width / 2.0f - rot_center.x;
			rotation_mat.at<double>(1, 2) += bbox.height / 2.0f - rot_center.y;
	
			cv::Mat dst;
			cv::warpAffine(src_img, dst, rotation_mat, bbox.size());

			cv::Mat NCC_Output;
			cv::matchTemplate(dst,avg_plane[i],NCC_Output,cv::TM_CCOEFF_NORMED);
			
			double maxVal, minVal;
			cv::Point minP, maxP;
			cv::minMaxLoc(NCC_Output,&minVal,&maxVal, &minP, &maxP);
			//Review this carefully (for debug purposes)
			/*
			for (int y = 0; y < NCC_Output.rows; y++)
			{
				float* yRow = NCC_Output.ptr<float>(y);
				for (int x = 0; x < NCC_Output.cols; x++)
				{
					yRow[x] = ( (yRow[x]-minVal)/(maxVal-minVal))*255;
				}
			}
			*/
			//NCC_Output.convertTo(NCC_Output,CV_8U);
			std::cout << "NCC_output.cols " << NCC_Output.cols << " NCC_output.rows " << NCC_Output.rows << "\n";
	    	//ipa::imshow("Template matching result image", NCC_Output, true, 0.27f);
			
			cvtColor(dst,dst,cv::COLOR_GRAY2BGR);

			// Calculate the center of the matched region
			cv::Point matchCenter(maxP.x + avg_plane[i].cols / 2, maxP.y + avg_plane[i].rows / 2);

			// Draw a rectangle around the matched region(for visualization)
	    	cv::rectangle(dst, maxP, cv::Point(maxP.x + avg_plane[i].cols, maxP.y + avg_plane[i].rows), cv::Scalar(0, 255, 0));
			//cv::circle(image, matchCenter, 5, cv::Scalar::all(0), 2, 8, 0);
			cv::circle(dst,matchCenter,20,cv::Scalar(0,0,255),cv::FILLED);
	    	//cv::circle(dst, maxP, 25, cv::Scalar(0, 255, 0), cv::FILLED);
			std::cout << "dst.cols " << dst.cols << " dst.rows " << dst.rows << "\n";
			ipa::imshow("Rotated image",dst,false,0.27f);
		    cv::waitKey(300);
		}
		std::cout << "Done template " << i << "\n";
	}
	
	/*
	// REVIEW CAREFULLY FROM THIS POINT ONWARDS
	//----------------------------------------------

	//cv::Mat src_img = cv::imread(std::string(DATASET_PATH)+"/488_DT8.jpg",cv::IMREAD_GRAYSCALE);

	std::vector<cv::Rect> yolo_boxes; // yolo boxes in the original image
	const std::string yolo_label_path = std::string(DATASET_PATH)+ "/488_DT8.txt";
	processYoloLabels(yolo_label_path, src_img, yolo_boxes);
	std::vector<std::vector<cv::RotatedRect>> rotated_yolo_boxes;
	std::vector<cv::Mat> rotated_images;
    for( int degree_angle = 0; degree_angle < 360; degree_angle += 5)
	{

	
		cv::Point rot_center = cv::Point(ucas::round<float>(src_img.cols / 2.0f),
									     ucas::round<float>(src_img.rows / 2.0f));

		cv::Mat rotation_mat = cv::getRotationMatrix2D(rot_center, degree_angle, 1);

		cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src_img.size(), degree_angle).boundingRect2f();

		// adjust transformation matrix
		rotation_mat.at<double>(0, 2) += bbox.width / 2.0f - rot_center.x;
		rotation_mat.at<double>(1, 2) += bbox.height / 2.0f - rot_center.y;

		cv::Mat dst;
		cv::warpAffine(src_img, dst, rotation_mat, bbox.size());
		rotated_images.push_back(dst);
		

		for(const auto& yolo_box : yolo_boxes) 
		{
			cv::Point2f box_center(yolo_box.x + yolo_box.width /2.0f, yolo_box.y + yolo_box.height /2.0f);
	
	        // Create a column 3x1 vector for the box center so that we can apply the rotation
			// because getRotationMatrix2D expects a 2x3 matrix
	        cv::Mat box_center_mat(cv::Size(1,3), CV_64FC1);
			box_center_mat.at<double>(0, 0) = box_center.x;
			box_center_mat.at<double>(1, 0) = box_center.y;
            box_center_mat.at<double>(2, 0) = 1;

			// Apply the rotation: here rotated_box_center_mat is a 2x1 matrix
			cv::Mat rotated_box_center_mat = rotation_mat * box_center_mat;

			// Convert back to cv::Point2f
			cv::Point2f rotated_box_center(rotated_box_center_mat.at<double>(0,0), rotated_box_center_mat.at<double>(1,0));
   

			// Applico la correzione per tenere conto del fatto che gli yolo boxes ruotati si trovano in dst
			// che è l'immagine ruotata e pertanto ha un'origine diversa rispetto all'immagine scevra da rotazione (src_img)
            rotated_box_center.x += bbox.width /2.0f - rot_center.x;
			rotated_box_center.y += bbox.height /2.0f- rot_center.y;

			cv::RotatedRect rotated_box(rotated_box_center, cv::Size2f(yolo_box.width, yolo_box.height), degree_angle);

			rotated_yolo_boxes[degree_angle/5].push_back(rotated_box);
        }

    }
	std::vector<std::string>avg_airplanes_paths;
	globFiles(avg_airplanes_dir.string(), "/*.png", avg_airplanes_paths);

	std::vector<cv::Mat> avg_plane1;
	for (const auto& path : avg_airplanes_paths)
	{
		cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
		if (img.data)
			avg_plane1.push_back(img);
	}

	const auto num_avg_airplanes_new = avg_plane1.size();

	std::vector<cv::Point> positive_samples;
	std::vector<cv::Point> negative_samples;
	for (int i = 0; i < num_avg_airplanes; i++)
	{


			for(size_t j=0; j < rotated_images.size(); j++)
			{
				cv::Mat NCC_Output;
				cv::matchTemplate(rotated_images[j],avg_plane[i],NCC_Output,cv::TM_CCOEFF_NORMED);
			

				double maxVal, minVal;
				cv::Point minP, maxP;
				cv::minMaxLoc(NCC_Output, &minVal, &maxVal, &minP, &maxP);

			    // CONTOLLARE SE I PUNTI A MASSIMA CORR SONO CONTENUTI NELLE YOLO DELLA J-esima immagine ruotata
				//rotated_yolo_boxes[j] // j-esimo vettore contente gli yolo boxes ruotati della j-esima immagine ruotata

				//in the case of multi template matching
			    double threshold = 0.9;
				std::vector<cv::Point> maxCorrPoints;
				maxCorrPointPool(maxCorrPoints,NCC_Output,threshold);

				// for standard template matching
				//pointInBoxes(maxP, rotated_yolo_boxes[j]) ? positive_samples.push_back(maxP) : negative_samples.push_back(maxP); 	
				
				//for multi template matching
				for(const auto& point : maxCorrPoints)
					pointInBoxes(point, rotated_yolo_boxes[j]) ? positive_samples.push_back(point) : negative_samples.push_back(point);
				
				selected_points();


			}

	}
	*/

/*
 * FEATURES EXTRACTION
 */ 
 //std::map<std::string, std::vector<cv::Mat>> samples_for_svm;
// 
// (key,value) TP, vettore di tutti i TP di tutte le immagini ruotate
// (key,value) FP, vettore di tutti i FP di tutte le immagini ruotate

// void featuresExtraction(std::map<std::string, std::vector<std::vector <cv::Mat>> >& samples_for_svm);

	//----------------------------------------------



	return EXIT_SUCCESS;
}

/*
bool pointInBoxes(const cv::Point& point, const std::vector<cv::RotatedRect>& boxes)
{
	for (const auto& box : boxes) {
		if (box.contains(point)) {
			return true;
		}
	}
	return false;
}
*/
void maxCorrPointPool(std::vector<cv::Point>& maxCorrPoints,const cv::Mat& NCC_output,const double threshold)
{
	
	for(int y = 0; y < NCC_output.rows; y++)
	{
		const float* yRow = NCC_output.ptr<float>(y);
		for(int x = 0; x < NCC_output.cols; x++)
		{
			if(yRow[x] > threshold)
				maxCorrPoints.emplace_back(x,y);
			
		}
	}
	
}
bool pointInBoxes(const cv::Point2f& point, const std::vector<cv::RotatedRect>& boxes)
{
	for (const auto& box : boxes) 
	{
		cv::Point2f vertices[4];
		box.points(vertices);
		const std::vector<cv::Point2f> contour(vertices, vertices + 4);
		if (cv::pointPolygonTest(contour, point, false) >= 0 ? true : false)
			return true;
	}
	return false;
}

double euclidean_distance(const cv::Point& a,const cv::Point& b)
{
	return cv::norm(a - b);
}

std::vector<cv::Point> selected_points(std::vector<cv::Point>& points, double min_distance)
{
	// Sort the points by their coordinates
	std::sort(points.begin(), points.end(),
		[](const cv::Point& a, const cv::Point& b) {
			return a.x < b.x || (a.x == b.x && a.y < b.y);
		}
	);

	std::vector<cv::Point> selected_points;
	for(const cv::Point& point : points)
	{
		bool too_close = false;

		//Check if the point is too close to any previously selected points
		for(const cv::Point& selected_point : selected_points)
		{
			if (euclidean_distance(point, selected_point) < min_distance)
			{
				too_close = true;
				break;
			}
		}


		// If the point is not too close to any previously selected points, select it
		if(!too_close)
			selected_points.push_back(point);

	}

	return selected_points;
}
