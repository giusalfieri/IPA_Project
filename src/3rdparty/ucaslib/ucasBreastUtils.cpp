#include "ucasBreastUtils.h"
#include "ucasTypes.h"

// returns the binary image consisting of the segmented breast
cv::Mat ucas::breastSegment(
	const cv::Mat &image,			// input mammogram (either 8- or 16-bit grayscale image)
	ucas::binarizationMethod method,// binarization method
	bool noBlack /*= false*/,		// exclude black (=0) pixels from computation of global threshold
	bool noWhite /*= false*/,		// exclude white (=2^depth-1) pixels from computation of global threshold
	bool bracket_histo /*= false*/,	// bracket the histogram to the range that holds data to make it quicker
	ucas::StackPrinter *printer)
	
{
	// checks
	if(!image.data)
		UCAS_THROW("in breastSegment(): invalid image");
	if(image.channels() != 1)
		UCAS_THROW("in breastSegment(): unsupported number of channels");
	if(image.depth() != CV_8U && image.depth() != CV_16U)
		UCAS_THROW("in breastSegment(): unsupported bitdepth: only 8- and 16-bit grayscale image are supported");

	// create a clone of the given image
	cv::Mat res = image.clone();

	// calculate histogram
	std::vector<int> histo = histogram(image);

	// histogram manipulations prior to binarization
	int threshold_shift=0;
	if(noBlack)
		histo[0] = 0;
	if(noWhite)
		histo[histo.size()-1]=0;
	if(bracket_histo)
		histo = compressHistogram(histo, threshold_shift);

	// apply selected binarization method
	if(method == ucas::otsuopencv)
	{
		// convert to 8 bit if image is 16 bit, since OpenCV thresholding functions can be applied to 8 bit images only
		if(res.depth() == CV_16U)
			res.convertTo(res, CV_8U, 255.0/65535.0);
		cv::threshold(res, res, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	}
	else if(method == ucas::otsu)
		ucas::binarize(res, getOtsuAutoThreshold(histo)+threshold_shift);
	else if(method == ucas::isodata)
		ucas::binarize(res, getIsoDataAutoThreshold(histo)+threshold_shift);
	else if(method == ucas::mean)
		ucas::binarize(res, getMeanThreshold(histo)+threshold_shift);
	else if(method == ucas::minerror)
		ucas::binarize(res, getMinErrorIThreshold(histo)+threshold_shift);
	else if(method == ucas::maxentropy)
		ucas::binarize(res, getMaxEntropyAutoThreshold(histo)+threshold_shift);
	else if(method == ucas::renyientropy)
		ucas::binarize(res, getRenyiEntropyAutoThreshold(histo)+threshold_shift);
	else if(method == ucas::yen)
		ucas::binarize(res, getYenyAutoThreshold(histo)+threshold_shift);
	else if(method == ucas::triangle)
		ucas::binarize(res, getTriangleAutoThreshold(histo)+threshold_shift);
	else if(method == ucas::all)
	{
		// convert to 8 bit if image is 16 bit, since OpenCV thresholding functions can be applied to 8 bit images only
		if(res.depth() == CV_16U)
			res.convertTo(res, CV_8U, 255.0/65535.0);
		cv::threshold(res, res, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		if(!ucas::checkBreastMask(res))
		{
			if(printer)
				printer->printf("Otsu failed, try Yeni\n");
			res = image.clone();
			ucas::binarize(res, getYenyAutoThreshold(histo)+threshold_shift);
			if(!ucas::checkBreastMask(res))
			{
				if(printer)
					printer->printf("Yeni failed, try Renyi\n");
				res = image.clone();
				ucas::binarize(res, getRenyiEntropyAutoThreshold(histo)+threshold_shift);
				if(!ucas::checkBreastMask(res))
				{
					if(printer)
						printer->printf("Renyi failed, try MaxEntropy\n");
					res = image.clone();
					ucas::binarize(res, getMaxEntropyAutoThreshold(histo)+threshold_shift);
					if(!ucas::checkBreastMask(res))
					{
						if(printer)
							printer->printf("MaxEntropy failed, try MinError\n");
						res = image.clone();
						ucas::binarize(res, getMinErrorIThreshold(histo)+threshold_shift);

						if(!ucas::checkBreastMask(res))
							UCAS_THROW("cannot segment breast: all binarization methods failed");
					}
				}
			}
		}
	}
	else
		UCAS_THROW("in breastSegment(): unsupported binarization method");


	// closing
	//cv::morphologyEx(res, res, CV_MOP_CLOSE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(60,60)));

	// select greatest connected component
	std::vector< std::vector<cv::Point> > ccs;
	cv::findContours(res, ccs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	double areaMax = -1;
	int ccsIdxAreaMax = -1;
	for(int i=0; i<ccs.size(); i++)
	{
		if(cv::contourArea(ccs[i]) > areaMax)
		{
			areaMax = cv::contourArea(ccs[i]);
			ccsIdxAreaMax = i;
		}
	}
	if(ccsIdxAreaMax != -1)
	{
		std::vector< std::vector<cv::Point> > ccsAreaMax;
		ccsAreaMax.push_back(ccs[ccsIdxAreaMax]);
		res.setTo(cv::Scalar(0));
		cv::drawContours(res, ccsAreaMax, -1, cv::Scalar(255), cv::FILLED);
	}

	return res;
}

//returns true if the given binary image satisfies some necessary conditions to contain a breast sagoma
bool ucas::checkBreastMask(
	const cv::Mat & mask,			// breast mask
	float minF,						// minimum fraction of white/total pixels
	float maxF						// maximum fraction of white/total pixels
	) 
{
#if DM_VERBOSE>3
	printf("\t\t\t\tin SampleExtractor::breastIsSagoma(...)\n");
#endif

	if(mask.depth() != CV_8U)
		UCAS_THROW("cannot check breast binary mask: unsuported bit depth (only 8-bit is supported)");

	//counting black and white pixels
	int black_c=0, white_c=0;
	for(int y=0; y<mask.rows; y++)
	{
		const ucas::uint8* row_ptr = mask.ptr<ucas::uint8>(y);
		for(int x=0; x<mask.cols; x++)
			if(row_ptr[x])
				white_c++;
			else
				black_c++;
	}

	// can't be all black or all white
	if(white_c == 0 || black_c == 0)
		return false;

	// segmented area should be within a certain range of the total area of the image
	if(float(white_c) < minF*mask.rows*mask.cols || float(white_c) > maxF*mask.rows*mask.cols)
		return false;

	return true;
}