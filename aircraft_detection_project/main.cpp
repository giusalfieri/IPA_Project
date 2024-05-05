// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"


int main()
{

	cv::Mat img = cv::imread(std::string(DATASET_PATH)+"/488_DT8.jpg");


	ipa::imshow("Original image", img);


	return EXIT_SUCCESS;
}

