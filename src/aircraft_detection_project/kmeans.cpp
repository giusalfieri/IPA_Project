#include "kmeans.h"
#include "utils.h"


/**
 * @brief Perform K-Means clustering on extracted templates based on their dimensions.
 *
 * This function clusters extracted templates using their width and height.
 *
 * @param extracted_templates A vector of cv::Mat objects representing the extracted templates.
 * @param K The number of clusters to form.
 * @return cv::Mat A matrix of labels indicating the cluster for each template.
 *
 * The function works as follows:
 * 1. Creates a matrix to store the dimensions of each template.
 * 2. Fills this matrix with the width and height of each template.
 * 3. Performs K-Means clustering on the dimensions matrix.
 * 4. Returns the cluster labels.
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
 * @brief Perform K-Means clustering on templates based on their mean intensity.
 *
 * This function clusters templates by computing their mean intensity and applying K-Means clustering.
 *
 * @param clustered_templates_by_size A vector of cv::Mat objects representing the templates to be clustered.
 * @param K_clusters The number of intensity clusters to form.
 * @return cv::Mat A matrix of labels indicating the cluster assignment for each template.
 *
 * The function works as follows:
 * 1. Creates a matrix to store the mean intensity of each template.
 * 2. Fills this matrix with the mean intensity of each template.
 * 3. Performs K-Means clustering on the intensity matrix.
 * 4. Returns the cluster labels.
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
 * @brief Save images to corresponding cluster directories based on clustering labels.
 *
 * This function saves images to specific directories based on their assigned cluster labels.
 *
 * @param images A vector of cv::Mat objects representing the images to be saved.
 * @param image_paths A vector of strings representing the original file paths of the images.
 * @param labels A cv::Mat object containing the cluster labels for each image.
 * @param cluster_paths A vector of std::filesystem::path objects representing the directories for each cluster.
 *
 * The function works as follows:
 * 1. Iterates over each image in the provided vector.
 * 2. Retrieves the cluster label for each image.
 * 3. Constructs the output file path for the clustered image based on its original file name and cluster directory.
 * 4. Saves the image to the constructed file path with a ".png" extension.
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