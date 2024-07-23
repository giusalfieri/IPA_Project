#include "kmeans.h"
#include "utils.h"


/**
 * @brief Performs K-Means clustering on a set of templates based on their dimensions.
 *
 * This function takes a vector of extracted templates and performs K-Means clustering
 * based on their width and height. The result is a matrix of labels indicating the
 * cluster assignment for each template.
 *
 * @param[in] extracted_templates A vector of `cv::Mat` objects representing the extracted templates.
 * @param[in] K The number of clusters to form.
 * @return A `cv::Mat` containing the cluster labels for each template.
 *
 * @note The function creates a matrix with the dimensions of each template and uses
 *       this matrix as input for the K-Means clustering algorithm.
 *
 * @see cv::kmeans
 */
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


/**
 * @brief Performs K-Means clustering on a set of templates based on their mean intensity.
 *
 * This function takes a vector of templates that have been clustered by size and performs
 * K-Means clustering based on their mean intensity. The result is a matrix of labels indicating
 * the cluster assignment for each template.
 *
 * @param[in] clustered_templates_by_size A vector of `cv::Mat` objects representing the templates
 *            that have been previously clustered by size.
 * @param[in] K_clusters The number of intensity-based clusters to form.
 * @return A `cv::Mat` containing the cluster labels for each template based on intensity.
 *
 * @note The function calculates the mean intensity of each template and uses this information
 *       as input for the K-Means clustering algorithm.
 *
 * @see cv::mean
 * @see cv::kmeans
 */
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

	// K-Means Clustering 
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


/**
 * @brief Saves images into directories based on their cluster labels.
 *
 * This function takes a vector of images, their corresponding file paths, cluster labels,
 * and destination directories for each cluster. It saves each image into the appropriate
 * directory based on its cluster label.
 *
 * @param[in] images A vector of `cv::Mat` objects representing the images to be saved.
 * @param[in] image_paths A vector of strings containing the original file paths of the images.
 * @param[in] labels A `cv::Mat` containing the cluster labels for each image.
 * @param[in] cluster_paths A vector of `std::filesystem::path` objects representing the destination
 *            directories for each cluster.
 *
 * @note The function assumes that the number of images, file paths, and labels are the same.
 *       Each image is saved with its original filename into the directory corresponding to its cluster label.
 *
 * @see cv::imwrite
 * @see std::filesystem::path
 */
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