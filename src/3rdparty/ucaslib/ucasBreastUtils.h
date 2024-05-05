#ifndef _UCAS_BREAST_UTILS
#define _UCAS_BREAST_UTILS

#include "ucasImageUtils.h"
#include "ucasLog.h"

/*****************************************************************
*   Utility methods              								 *
******************************************************************/
namespace ucas
{
	// returns the binary image consisting of the segmented breast
	cv::Mat breastSegment(
		const cv::Mat & image,			// input mammogram (either 8- or 16-bit grayscale image)
		binarizationMethod method = all,// binarization method (brute force attack if not defined)
		bool noBlack = false,			// exclude black (=0) pixels from computation of global threshold
		bool noWhite = false,			// exclude white (=2^depth-1) pixels from computation of global threshold
		bool bracket_histo = true,		// bracket the histogram to the range that holds data to make it quicker
		ucas::StackPrinter *printer = 0)
		;

	// returns true if the given breast mask contains B white pixels, with minP*size(mask) <= B <= maxP*size(mask)
	bool checkBreastMask(
		const cv::Mat & mask,			// breast mask
		float minF = 0.05,				// minimum fraction of white/total pixels
		float maxF = 0.95				// maximum fraction of white/total pixels
	) ;
}


/*****************************************************************
*   Data structures              								 *
******************************************************************/
namespace ucas
{
	const int MAMMO_MAX_SIZE = 8000;					//maximum size (in pixels, per side) of mammograms

	struct breastConvexArea
	{
		unsigned short *x_start, *x_end;
		unsigned short y_start, y_end;

		breastConvexArea(void)
		{
			x_start = new unsigned short [MAMMO_MAX_SIZE];
			x_end = new unsigned short [MAMMO_MAX_SIZE];
			for(int i=0; i<MAMMO_MAX_SIZE; i++)
				x_start[i] = x_end[i] = -1;
			y_start = y_end = -1;
		}
	};
}

#endif