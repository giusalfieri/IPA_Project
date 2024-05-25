#include "template_extraction.h"
#include "kmeans.h"
#include "utils.h"




int main()
{
	/*
    ------------------------------------------
	**** TEMPLATE EXTRACTION FROM DATASET ****
	------------------------------------------
	*/
    //extractTemplates();
	//----------------------------------------


	/*
	------------------------------------------
	*********  K-MEANS  CLUSTERING  **********  
	------------------------------------------
	*/

	/*
	--------------------------
	1)  K-MEANS BY IMAGES SIZE     
    --------------------------
	*/

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
	*/
	//--------------------


	//------------------------
	// 1.2) Performing K-Means
	//------------------------
	const int num_clusters_by_size =  3;
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
	*/

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
	
	
	int dim = 0;
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

		reshape2sameDim(intensities_img);



		for(int j=0; j < intensities_img.size(); j++)
		{
			const auto  final_template_id = std::filesystem::path(final_clusters_paths[j]).stem();
			const std::filesystem::path clusterd_by_intensity_template_path = ith_cluster_by_intensity_folder_path / final_template_id;

			// Save the template image to the corresponding cluster folder
			cv::imwrite(clusterd_by_intensity_template_path.string() + ".png", intensities_img[j]);
		}

	}


	/*
	for (const auto& intensity_cluster_path : final_clusters_by_intensity_paths)
	{
	
		std::vector<std::string>final_clusters_paths;
		globFiles(intensity_cluster_path.string(), "/*.png", final_clusters_paths, true);

		std::vector<cv::Mat> intensities_img;
		for (const auto& path : final_clusters_paths)
		{
			cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
			if (img.data)
				intensities_img.push_back(img);

			dim = img.rows;
		}

		reshape2sameDim(intensities_img);


		for (auto& img : intensities_img)
		{

			cv::imwrite(, img);
		}
	}
	*/



	/*
	std::vector<std::string>final_clusters_paths;
	globFiles(kmeans_intensity_folder_path, "/*.png", final_clusters_paths, true);
	int dim = 0;
	std::vector<cv::Mat> intensities_img;
	for (const auto& path : final_clusters_paths)
	{
		cv::Mat img = cv::imread(path);
		if (img.data)
			intensities_img.push_back(img);

		dim = img.rows;
	}

	const int num_clusters_by_intensity = 15;
	std::vector<cv::Mat> average_airplanes;

	for (int k = 0; k < num_clusters_by_intensity; k++)
	{
		cv::Mat img = eigenPlanes(intensities_img, dim);
		average_airplanes.push_back(img);

	}

	for (const auto& avg_airplane : average_airplanes)
	{
		ipa::imshow("Average Airplane", avg_airplane, true);
	}
	*/
	return EXIT_SUCCESS;
}

