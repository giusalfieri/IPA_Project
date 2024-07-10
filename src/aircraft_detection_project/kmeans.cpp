#include "kmeans.h"
#include "utils.h"


cv::Mat kmeansBySize(const std::vector<cv::Mat>& extracted_templates, int K)
{
	// Creation of a matrix of 'templates.size()' rows and 2 columns
	// This matrix will store the dimensions (width, height) of each extracted template
	cv::Mat extracted_templates_dims(extracted_templates.size(), 2, CV_32F);

	// Fill the sizes matrix with the dimensions of each extracted template
	for (int i = 0; i < extracted_templates.size(); i++)
	{
		float* yRow = extracted_templates_dims.ptr<float>(i);
		yRow[0] = extracted_templates[i].cols;  // the first column stores the width  of the i-th template
		yRow[1] = extracted_templates[i].rows;  // the first column stores the height of the i-th template
	}

	// K-Means Clustering 
	cv::Mat labels, centers;
	cv::kmeans(
		extracted_templates_dims, 
		K, 
		labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 
		50, 
		cv::KMEANS_PP_CENTERS,
		centers);

	return labels;	
}



cv::Mat kmeansByIntensity(const std::vector<cv::Mat>& clustered_templates_by_size, int K_clusters)
{
	cv::Mat intensities(clustered_templates_by_size.size(), 1, CV_32F);

	for (int i = 0; i < clustered_templates_by_size.size(); i++)
	{
		cv::Scalar mean_intensity = cv::mean(clustered_templates_by_size[i]);

		float* yRow = intensities.ptr<float>(i);
		yRow[0] = static_cast <float> (mean_intensity[0]);
	}

	cv::Mat labels, centers;

	cv::kmeans(
		intensities,
		K_clusters,
		labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1),
		100,
		cv::KMEANS_PP_CENTERS,
		centers
	);

	return labels;
}

void saveClusteredImages(const std::vector<cv::Mat>& images,const std::vector<std::string>& image_paths, const cv::Mat& labels,const std::vector<std::filesystem::path>& cluster_paths)
{
	for (int i = 0; i < images.size(); i++) 
	{
		int cluster_id = labels.at<int>(i);
		const auto image_id = std::filesystem::path(image_paths[i]).stem();
		const std::filesystem::path clustered_image_path = cluster_paths[cluster_id] / image_id;
		cv::imwrite(clustered_image_path.string() + ".png", images[i]);
	}
}