/*
    Template for k-mean
    TODO -> filesystem for image upload 
*/

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


