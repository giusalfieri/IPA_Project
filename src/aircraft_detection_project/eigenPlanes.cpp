#include "eigenPlanes.h"
#include "utils.h"

cv::Mat createDataMatrix(const std::vector<cv::Mat>& images);


cv::Mat eigenPlanes(const std::vector<cv::Mat>& vec, const int img_dim)
{
	cv::Mat data = createDataMatrix(vec);

	cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.95);

	cv::Mat avg_airplane = pca.mean.reshape(1, img_dim);


	return avg_airplane;
}


cv::Mat createDataMatrix(const std::vector<cv::Mat>& images)
{
	cv::Mat data;
	for (const cv::Mat& img : images)
	{
		cv::Mat imgVector = img.reshape(1, 1);
		data.push_back(imgVector);
	}

	return data;
}
