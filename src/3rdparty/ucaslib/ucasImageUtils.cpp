#include "ucasImageUtils.h"
#include "ucasStringUtils.h"
#include "ucasFileUtils.h"
#include "ucasMathUtils.h"
#include "ucasLog.h"
#include "ucasTypes.h"

#ifdef WITH_GDCM
#include "gdcmImage.h"
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#endif

namespace
{
	// image histogram statistics
	inline double A(const std::vector<int> &y, int j=-1) 
	{
		if(j < 0 || j >= y.size())
			j = int(y.size())-1;
		double x = 0;
		for (int i=0;i<=j;i++)
			x+=y[i];
		return x;
	}
	inline double B(const std::vector<int> &y, int j=-1) 
	{
		if(j < 0 || j >= y.size())
			j = int(y.size())-1;
		double x = 0;
		for (int i=0;i<=j;i++)
			x+=i*y[i];
		return x;
	}
	inline double C(const std::vector<int> &y, int j=-1) 
	{
		if(j < 0 || j >= y.size())
			j = int(y.size())-1;
		double x = 0;
		for (int i=0;i<=j;i++)
			x+=i*i*y[i];
		return x;
	}
}

// converts the OpenCV depth flag into the corresponding bitdepth
int ucas::imdepth(int ocv_depth)
{
	switch(ocv_depth)
	{
		case CV_8U:  return 8;
		case CV_8S:  return 8;
		case CV_16U: return 16;
		case CV_16S: return 16;
		case CV_32S: return 32;
		case CV_32F: return 32;
		case CV_64F: return 64;
		default:     return -1;
	}
}

// detects bitdepth from maximum value of the image
int	ucas::imdepth_detect(const cv::Mat & img)
{
	double min=0, max=0;
	cv::minMaxLoc(img, &min, &max);
	return static_cast<int>(std::ceil(ucas::log2<double>(max)));
}

// extends the OpenCV "imshow" function with the addition of a scale factor
void ucas::imshow(const std::string winname, cv::InputArray arr, bool wait, float scale)
{
	// create window
	cv::namedWindow(winname, cv::WINDOW_FREERATIO);

	// resize window so as to fit screen size while maintaining image aspect ratio
	int win_height = arr.size().height, win_width = arr.size().width;

	cv::resizeWindow(winname, ucas::round(win_width*scale), ucas::round(win_height*scale));

	// display image
	cv::imshow(winname, arr);

	// wait for key pressed
	if(wait)
		cv::waitKey(0);
}

cv::Mat ucas::imread(const std::string & path, int flags, int *bits_used) 
{
	// check file exists first
	if(!ucas::isFile(path))
		throw ucas::FileNotExistsError(path);

	// create cv::Mat
	cv::Mat mat;

	// check for DICOM extension and try to load with GDCM library, if present
	if(ucas::getFileExtension(path) == "dcm" || ucas::getFileExtension(path) == "DCM")
	{
#ifdef WITH_GDCM

		// read file
		gdcm::ImageReader imreader;
		imreader.SetFileName( path.c_str() );
		if(!imreader.Read())
			UCAS_THROW("GDCM cannot read DICOM image");

		// get image
		gdcm::Image &image = imreader.GetImage();

		// check number of dimensions
		if(image.GetNumberOfDimensions() != 2)
			UCAS_THROW(ucas::strprintf("GDCM bridge cannot read %dD DICOM image: feature not yet implemented", image.GetNumberOfDimensions()));

		// checks preconditions
		if(image.GetPixelFormat().GetBitsAllocated() != 8 && image.GetPixelFormat().GetBitsAllocated() != 16)
			UCAS_THROW(ucas::strprintf("Unsupported DICOM image: bits allocated are %d, but we currently support only 8 and 16", image.GetPixelFormat().GetBitsAllocated()));
		if(image.GetPixelFormat() != gdcm::PixelFormat::UINT8 && image.GetPixelFormat() != gdcm::PixelFormat::UINT16)
			UCAS_THROW(ucas::strprintf("Unsupported DICOM image: pixel format is %s, but we currently support only UINT8 and UINT16", image.GetPixelFormat().GetScalarTypeAsString()));
		if(image.GetPixelFormat().GetSamplesPerPixel() != 1)
			UCAS_THROW(ucas::strprintf("Unsupported DICOM image: samples per pixel are %d, but we currently support only 1", image.GetPixelFormat().GetSamplesPerPixel()));
		
		// create data
		mat = cv::Mat(image.GetRows(), image.GetColumns(), image.GetPixelFormat().GetBitsAllocated() == 8 ? CV_8U : CV_16U);

		// read data
		if(!image.GetBuffer(reinterpret_cast<char*>(mat.data)))
			UCAS_THROW("GDCM cannot read DICOM pixel data");

		if(bits_used)
			*bits_used = image.GetPixelFormat().GetBitsStored();
#else
		UCAS_THROW("Cannot read DICOM file: GDCM library not found. Please re-configure the build from source and enable GDCM.");
#endif
	}
	else if(ucas::getFileExtension(path) == "cvmat" || ucas::getFileExtension(path) == "CVMAT")
	{
		FILE *f = fopen(path.c_str(), "rb");
		if(!f)
			throw CannotOpenFileError(path);
		int channels = 0, _depth = 0, rows = 0, cols = 0;
		if(!fread(&rows, sizeof(int), 1, f))
			throw Error("Cannot read #rows from the .cvmat");
		if(!fread(&cols, sizeof(int), 1, f))
			throw Error("Cannot read #cols from the .cvmat");
		if(!fread(&channels, sizeof(int), 1, f))
			throw Error("Cannot read #channels from the .cvmat");
		if(!fread(&_depth,    sizeof(int), 1, f))
			throw Error("Cannot read depth from the .cvmat");
		size_t imgsize = size_t(rows) * cols * channels * ucas::imdepth(_depth) / 8;

		// _depth correction for multi(>3)channel images
		if(channels > 1)
		{
			if(_depth == CV_8U)
				_depth = CV_8UC(channels);
			else if(_depth == CV_16U)
				_depth = CV_16UC(channels);
			else if(_depth == CV_32F)
				_depth = CV_32FC(channels);
			else if(_depth == CV_64F)
				_depth = CV_64FC(channels);
			else
				throw Error("Unsupported data type for multichannel image");
		}
		mat = cv::Mat(rows, cols, _depth);
		if(!fread(mat.data,  imgsize, 1, f))
			throw Error("Cannot read data from the .cvmat");
		fclose(f);

		if(bits_used)
			*bits_used = ucas::imdepth(mat.depth());
	}
	else
	{
		mat = cv::imread(path, flags);
		if(bits_used)
			*bits_used = ucas::imdepth(mat.depth());
	}
	return mat;
}

// extends the OpenCV "imwrite" function by adding support to OpenCV mat (.cvmat) and grayscale 8-16 bits DICOM images (.dcm)
void ucas::imwrite(const std::string & path, cv::Mat & img) 
{
	// check for DICOM extension and try to load with GDCM library, if present
	if(ucas::getFileExtension(path) == "dcm" || ucas::getFileExtension(path) == "DCM")
	{

#ifdef WITH_GDCM

		// check image bitdepth
		if(img.depth() != CV_8U && img.depth() != CV_16U)
			UCAS_THROW("GDCM cannot generate DICOM image: only 8 and 16 bit grayscale images are supported");
		if(img.channels() != 1)
			UCAS_THROW("GDCM cannot generate DICOM image: only monochromatic images are supported");
		if(!img.isContinuous())
			UCAS_THROW("GDCM cannot generate DICOM image: only continuous images are supported");

		gdcm::ImageWriter writer;
		gdcm::Image &image = writer.GetImage();
		image.SetNumberOfDimensions( 2 );
		unsigned int dims[3] = {};
		dims[0] = img.rows;
		dims[1] = img.cols;
		image.SetDimensions( dims );
		gdcm::PixelFormat pf = img.depth() == CV_8U ? gdcm::PixelFormat::UINT8 : gdcm::PixelFormat::UINT16;
		pf.SetSamplesPerPixel( 1 );
		image.SetPixelFormat( pf );
		gdcm::PhotometricInterpretation pi = gdcm::PhotometricInterpretation::MONOCHROME2;
		image.SetPhotometricInterpretation( pi );
		image.SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );

		gdcm::DataElement pixeldata( gdcm::Tag(0x7fe0,0x0010) );
		unsigned long l = image.GetBufferLength();
		pixeldata.SetByteValue( reinterpret_cast<char*>(img.data), (uint32_t)l );
		image.SetDataElement( pixeldata );

		writer.SetFileName( path.c_str() );
		if( !writer.Write() )
			UCAS_THROW(ucas::strprintf("GDCM cannot write image at \"%s\"", path.c_str()));

#else
		UCAS_THROW("Cannot write DICOM file: GDCM library not found. Please re-configure the build from source and enable GDCM.");
#endif
	}
	else if(ucas::getFileExtension(path) == "cvmat" || ucas::getFileExtension(path) == "CVMAT")
	{
		if(!img.isContinuous())
			UCAS_THROW("Cannot save to .cvmat : only continuous images are supported");

		FILE *f = fopen(path.c_str(), "wb");
		if(!f)
			throw ucas::CannotWriteFileError(path);
		int channels = img.channels();
		int depth = img.depth();
		size_t imgsize = size_t(img.rows) * img.cols * channels * ucas::imdepth(img.depth()) / 8;
		fwrite(&img.rows, sizeof(int), 1, f);
		fwrite(&img.cols, sizeof(int), 1, f);
		fwrite(&channels, sizeof(int), 1, f);
		fwrite(&depth,    sizeof(int), 1, f);
		fwrite(img.data,  imgsize, 1, f);
		fclose(f);
	}
	else
		cv::imwrite(path, img);
}

// rescale the image from 'bits_in' to 'bits_out'
void ucas::imrescale(cv::Mat & image, int bits_in, int bits_out) 
{
	if(image.depth() == CV_8U)
	{
		if(bits_out > 8)
			UCAS_THROW(ucas::strprintf("Cannot rescale image: bits_out (%d) > stored bits (8)", bits_out));
		double f = (std::pow(2.0f, bits_out) - 1.0f) / ( std::pow(2.0f, bits_in) -1.0f);
		int max = int(std::pow(2.0f, bits_out) - 1);
		for(int y=0; y<image.rows; y++)
		{
			unsigned char* data = image.ptr<unsigned char>(y);
			for(int x=0; x<image.cols; x++)
				data[x] = (unsigned char)(std::min ( ucas::round( data[x] * f ), max ) );
		}
	}
	if(image.depth() == CV_16U)
	{
		if(bits_out > 16)
			UCAS_THROW(ucas::strprintf("Cannot rescale image: bits_out (%d) > stored bits (16)", bits_out));
		double f = (std::pow(2.0f, bits_out) - 1) /  ( std::pow(2.0f, bits_in)-1);
		int max = int(std::pow(2.0f, bits_out) - 1);
		for(int y=0; y<image.rows; y++)
		{
			unsigned short* data = image.ptr<unsigned short>(y);
			for(int x=0; x<image.cols; x++)
				data[x] = (unsigned short)(std::min ( ucas::round( data[x] * f ), max ) );
		}
	}
}

// return (x,y) offset of 'data' in parent image (if any), otherwise return (0,0)
cv::Point ucas::imOffsetInParent(const cv::Mat & img)
{
	cv::Size  wholesize;
	cv::Point offset;
	img.locateROI(wholesize, offset);
	return offset;
}

std::string ucas::binarizationMethod_toString(ucas::binarizationMethod code)
{
	if     (code == otsuopencv)		return "otsuopencv";
	else if(code == otsu)			return "otsu";
	else if(code == isodata)		return "isodata";
	else if(code == triangle)		return "triangle";
	else if(code == mean)			return "mean";
	else if(code == minerror)		return "minerror";
	else if(code == maxentropy)		return "maxentropy";
	else if(code == renyientropy)	return "renyientropy";
	else if(code == yen)			return "yen";
	else							return "all";
}
ucas::binarizationMethod ucas::binarizationMethod_toInt(const std::string & code)
{
	for(int i=0; i<9; i++)
		if(binarizationMethod_toString(ucas::binarizationMethod(i)).compare(code) == 0)
			return ucas::binarizationMethod(i);
	return ucas::binarizationMethod(all);
}
std::string ucas::binarizationMethods()
{
	std::string res = "{";
	for(int i=0; i<9; i++)
		res += "\"" + std::string(binarizationMethod_toString(binarizationMethod(i))) + "\"" + (i == 8 ? "}" : ", ");
	return res;
}

int ucas::getMeanThreshold(const std::vector<int> & data) 
{
	// C. A. Glasbey, "An analysis of histogram-based thresholding algorithms,"
	// CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.
	//
	// The threshold is the mean of the greyscale data
	return static_cast<int> ( std::floor(B(data)/A(data)) );
}

// Otsu's threshold algorithm
int ucas::getOtsuAutoThreshold(const std::vector<int> & data) 
{
	// Otsu's threshold algorithm
	// C++ code by Jordan Bevik <Jordan.Bevic@qtiworld.com>
	// ported to ImageJ plugin by G.Landini
	int k,kStar;  // k = the current threshold; kStar = optimal threshold
	int N1, N;    // N1 = # points with intensity <=k; N = total number of points
	double BCV, BCVmax; // The current Between Class Variance and maximum BCV
	double num, denom;  // temporary bookeeping
	int Sk;  // The total intensity for all histogram points <=k
	int S, L=int(data.size()); // The total intensity of the image

	// Initialize values:
	S = N = 0;
	for (k=0; k<L; k++){
		S += k * data[k];	// Total histogram intensity
		N += data[k];		// Total number of data points
	}

	Sk = 0;
	N1 = data[0]; // The entry for zero intensity
	BCV = 0;
	BCVmax=0;
	kStar = 0;

	// Look at each possible threshold value,
	// calculate the between-class variance, and decide if it's a max
	for (k=1; k<L-1; k++) { // No need to check endpoints k = 0 or k = L-1
		Sk += k * data[k];
		N1 += data[k];

		// The float casting here is to avoid compiler warning about loss of precision and
		// will prevent overflow in the case of large saturated images
		denom = (double)( N1) * (N - N1); // Maximum value of denom is (N^2)/4 =  approx. 3E10

		if (denom != 0 ){
			// Float here is to avoid loss of precision when dividing
			num = ( (double)N1 / N ) * S - Sk; 	// Maximum value of num =  255*N = approx 8E7
			BCV = (num * num) / denom;
		}
		else
			BCV = 0;

		if (BCV >= BCVmax){ // Assign the best threshold found so far
			BCVmax = BCV;
			kStar = k;
		}
	}
	// kStar += 1;	// Use QTI convention that intensity -> 1 if intensity >= k
	// (the algorithm was developed for I-> 1 if I <= k.)
	return kStar;
}

//Yen J.C., Chang F.J., and Chang S. (1995) 
// "A New Criterion for Automatic Multilevel Thresholding" IEEE Trans. on Image Processing, 4(3): 370-378
int ucas::getYenyAutoThreshold(const std::vector<int> & data) 
{
	int threshold;
	int ih, it;
	double crit;
	double max_crit;
	double * norm_histo = new double[data.size()]; /* normalized histogram */
	double * P1 = new double[data.size()]; /* cumulative normalized histogram */
	double * P1_sq = new double[data.size()];
	double * P2_sq = new double[data.size()];

	int total =0;
	for (ih = 0; ih < data.size(); ih++ )
		total+=data[ih];

	for (ih = 0; ih < data.size(); ih++ )
		norm_histo[ih] = (double)data[ih]/total;

	P1[0]=norm_histo[0];
	for (ih = 1; ih < data.size(); ih++ )
		P1[ih]= P1[ih-1] + norm_histo[ih];

	P1_sq[0]=norm_histo[0]*norm_histo[0];
	for (ih = 1; ih < data.size(); ih++ )
		P1_sq[ih]= P1_sq[ih-1] + norm_histo[ih] * norm_histo[ih];

	P2_sq[data.size() - 1] = 0.0;
	for ( ih = int(data.size())-2; ih >= 0; ih-- )
		P2_sq[ih] = P2_sq[ih + 1] + norm_histo[ih + 1] * norm_histo[ih + 1];

	/* Find the threshold that maximizes the criterion */
	threshold = -1;
	max_crit = -std::numeric_limits<double>::max();
	for ( it = 0; it < data.size(); it++ ) {
		crit = -1.0 * (( P1_sq[it] * P2_sq[it] )> 0.0? std::log( P1_sq[it] * P2_sq[it]):0.0) +  2 * ( ( P1[it] * ( 1.0 - P1[it] ) )>0.0? std::log(  P1[it] * ( 1.0 - P1[it] ) ): 0.0);
		if ( crit > max_crit ) {
			max_crit = crit;
			threshold = it;
		}
	}

	delete[] norm_histo;
	delete[] P1;
	delete[] P1_sq;
	delete[] P2_sq;

	return threshold;
}

// Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) 
// "A New Method for Gray-Level Picture Thresholding Using the Entropy of the Histogram" Graphical Models and Image Processing, 29(3): 273-285
int ucas::getRenyiEntropyAutoThreshold(const std::vector<int> & data) 
{
	int threshold; 
	int opt_threshold;

	int ih, it;
	int first_bin;
	int last_bin;
	int tmp_var;
	int t_star1, t_star2, t_star3;
	int beta1, beta2, beta3;
	double alpha;/* alpha parameter of the method */
	double term;
	double tot_ent;  /* total entropy */
	double max_ent;  /* max entropy */
	double ent_back; /* entropy of the background pixels at a given threshold */
	double ent_obj;  /* entropy of the object pixels at a given threshold */
	double omega;
	double * norm_histo = new double[data.size()]; /* normalized histogram */
	double * P1 = new double[data.size()]; /* cumulative normalized histogram */
	double * P2 = new double[data.size()];

	int total =0;
	for (ih = 0; ih < data.size(); ih++ )
		total+=data[ih];

	for (ih = 0; ih < data.size(); ih++ )
		norm_histo[ih] = (double)data[ih]/total;

	P1[0]=norm_histo[0];
	P2[0]=1.0-P1[0];
	for (ih = 1; ih < data.size(); ih++ ){
		P1[ih]= P1[ih-1] + norm_histo[ih];
		P2[ih]= 1.0 - P1[ih];
	}

	/* Determine the first non-zero bin */
	first_bin=0;
	for (ih = 0; ih < data.size(); ih++ ) {
		if ( !(ucas::abs(P1[ih])<2.220446049250313E-16)) {
			first_bin = ih;
			break;
		}
	}

	/* Determine the last non-zero bin */
	last_bin=int(data.size()) - 1;
	for (ih = int(data.size()) - 1; ih >= first_bin; ih-- ) {
		if ( !(ucas::abs(P2[ih])<2.220446049250313E-16)) {
			last_bin = ih;
			break;
		}
	}

	/* Maximum Entropy Thresholding - BEGIN */
	/* ALPHA = 1.0 */
	/* Calculate the total entropy each gray-level
	and find the threshold that maximizes it 
	*/
	threshold =0; // was MIN_INT in original code, but if an empty image is processed it gives an error later on.
	max_ent = 0.0;

	for ( it = first_bin; it <= last_bin; it++ ) {
		/* Entropy of the background pixels */
		ent_back = 0.0;
		for ( ih = 0; ih <= it; ih++ )  {
			if ( data[ih] !=0 ) {
				ent_back -= ( norm_histo[ih] / P1[it] ) * std::log ( norm_histo[ih] / P1[it] );
			}
		}

		/* Entropy of the object pixels */
		ent_obj = 0.0;
		for ( ih = it + 1; ih < data.size(); ih++ ){
			if (data[ih]!=0){
			ent_obj -= ( norm_histo[ih] / P2[it] ) * std::log ( norm_histo[ih] / P2[it] );
			}
		}

		/* Total entropy */
		tot_ent = ent_back + ent_obj;

		// IJ.log(""+max_ent+"  "+tot_ent);

		if ( max_ent < tot_ent ) {
			max_ent = tot_ent;
			threshold = it;
		}
	}
	t_star2 = threshold;

	/* Maximum Entropy Thresholding - END */
	threshold =0; //was MIN_INT in original code, but if an empty image is processed it gives an error later on.
	max_ent = 0.0;
	alpha = 0.5;
	term = 1.0 / ( 1.0 - alpha );
	for ( it = first_bin; it <= last_bin; it++ ) {
		/* Entropy of the background pixels */
		ent_back = 0.0;
		for ( ih = 0; ih <= it; ih++ )
			ent_back += std::sqrt ( norm_histo[ih] / P1[it] );

		/* Entropy of the object pixels */
		ent_obj = 0.0;
		for ( ih = it + 1; ih < data.size(); ih++ )
			ent_obj += std::sqrt ( norm_histo[ih] / P2[it] );

		/* Total entropy */
		tot_ent = term * ( ( ent_back * ent_obj ) > 0.0 ? std::log ( ent_back * ent_obj ) : 0.0);

		if ( tot_ent > max_ent ){
			max_ent = tot_ent;
			threshold = it;
		}
	}

	t_star1 = threshold;

	threshold = 0; //was MIN_INT in original code, but if an empty image is processed it gives an error later on.
	max_ent = 0.0;
	alpha = 2.0;
	term = 1.0 / ( 1.0 - alpha );
	for ( it = first_bin; it <= last_bin; it++ ) {
		/* Entropy of the background pixels */
		ent_back = 0.0;
		for ( ih = 0; ih <= it; ih++ )
			ent_back += ( norm_histo[ih] * norm_histo[ih] ) / ( P1[it] * P1[it] );

		/* Entropy of the object pixels */
		ent_obj = 0.0;
		for ( ih = it + 1; ih < data.size(); ih++ )
			ent_obj += ( norm_histo[ih] * norm_histo[ih] ) / ( P2[it] * P2[it] );

		/* Total entropy */
		tot_ent = term *( ( ent_back * ent_obj ) > 0.0 ? std::log(ent_back * ent_obj ): 0.0 );

		if ( tot_ent > max_ent ){
			max_ent = tot_ent;
			threshold = it;
		}
	}

	t_star3 = threshold;

	/* Sort t_star values */
	if ( t_star2 < t_star1 ){
		tmp_var = t_star1;
		t_star1 = t_star2;
		t_star2 = tmp_var;
	}
	if ( t_star3 < t_star2 ){
		tmp_var = t_star2;
		t_star2 = t_star3;
		t_star3 = tmp_var;
	}
	if ( t_star2 < t_star1 ) {
		tmp_var = t_star1;
		t_star1 = t_star2;
		t_star2 = tmp_var;
	}

	/* Adjust beta values */
	if ( std::abs ( t_star1 - t_star2 ) <= 5 )  {
		if ( std::abs ( t_star2 - t_star3 ) <= 5 ) {
			beta1 = 1;
			beta2 = 2;
			beta3 = 1;
		}
		else {
			beta1 = 0;
			beta2 = 1;
			beta3 = 3;
		}
	}
	else {
		if ( std::abs ( t_star2 - t_star3 ) <= 5 ) {
			beta1 = 3;
			beta2 = 1;
			beta3 = 0;
		}
		else {
			beta1 = 1;
			beta2 = 2;
			beta3 = 1;
		}
	}
	//IJ.log(""+t_star1+" "+t_star2+" "+t_star3);
	/* Determine the optimal threshold value */
	omega = P1[t_star3] - P1[t_star1];
	opt_threshold = (int) (t_star1 * ( P1[t_star1] + 0.25 * omega * beta1 ) + 0.25 * t_star2 * omega * beta2  + t_star3 * ( P2[t_star3] + 0.25 * omega * beta3 ));

	delete[] norm_histo;
	delete[] P1;
	delete[] P2;

	return opt_threshold;
}

// Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) 
// "A New Method for Gray-Level Picture Thresholding Using the Entropy of the Histogram" Graphical Models and Image Processing, 29(3): 273-285
int ucas::getMaxEntropyAutoThreshold(const std::vector<int> & data) 
{
	int threshold=-1;
	int ih, it;
	int first_bin;
	int last_bin;
	double tot_ent;  /* total entropy */
	double max_ent;  /* max entropy */
	double ent_back; /* entropy of the background pixels at a given threshold */
	double ent_obj;  /* entropy of the object pixels at a given threshold */
	double * norm_histo = new double[data.size()]; /* normalized histogram */
	double * P1 = new double[data.size()]; /* cumulative normalized histogram */
	double * P2 = new double[data.size()];

	int total =0;
	for (ih = 0; ih < data.size(); ih++ )
		total+=data[ih];

	for (ih = 0; ih < data.size(); ih++ )
		norm_histo[ih] = (double)data[ih]/total;

	P1[0]=norm_histo[0];
	P2[0]=1.0-P1[0];
	for (ih = 1; ih < data.size(); ih++ ){
		P1[ih]= P1[ih-1] + norm_histo[ih];
		P2[ih]= 1.0 - P1[ih];
	}

	/* Determine the first non-zero bin */
	first_bin=0;
	for (ih = 0; ih < data.size(); ih++ ) {
		if ( !(ucas::abs(P1[ih])<2.220446049250313E-16)) {
			first_bin = ih;
			break;
		}
	}

	/* Determine the last non-zero bin */
	last_bin=int(data.size()) - 1;
	for (ih = int(data.size()) - 1; ih >= first_bin; ih-- ) {
		if ( !(ucas::abs(P2[ih])<2.220446049250313E-16)) {
			last_bin = ih;
			break;
		}
	}

	// Calculate the total entropy each gray-level
	// and find the threshold that maximizes it 
	max_ent = -std::numeric_limits<double>::max();

	for ( it = first_bin; it <= last_bin; it++ ) {
		/* Entropy of the background pixels */
		ent_back = 0.0;
		for ( ih = 0; ih <= it; ih++ )  {
			if ( data[ih] !=0 ) {
				ent_back -= ( norm_histo[ih] / P1[it] ) * std::log ( norm_histo[ih] / P1[it] );
			}
		}

		/* Entropy of the object pixels */
		ent_obj = 0.0;
		for ( ih = it + 1; ih < data.size(); ih++ ){
			if (data[ih]!=0){
				ent_obj -= ( norm_histo[ih] / P2[it] ) * std::log ( norm_histo[ih] / P2[it] );
			}
		}

		/* Total entropy */
		tot_ent = ent_back + ent_obj;

		// IJ.log(""+max_ent+"  "+tot_ent);
		if ( max_ent < tot_ent ) {
			max_ent = tot_ent;
			threshold = it;
		}
	}

	delete[] norm_histo;
	delete[] P1;
	delete[] P2;

	return threshold;
}


// Kittler and J. Illingworth, "Minimum error thresholding," Pattern Recognition, vol. 19, pp. 41-47, 1986.
// C. A. Glasbey, "An analysis of histogram-based thresholding algorithms," CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.
int ucas::getMinErrorIThreshold(const std::vector<int> & data) 
{
	//Initial estimate for the threshold is found with the MEAN algorithm.
	int threshold =  getMeanThreshold(data); 
	int Tprev =-2;
	double mu, nu, p, q, sigma2, tau2, w0, w1, w2, sqterm, temp;
	while (threshold!=Tprev)
	{
		//Calculate some statistics.
		mu = B(data, threshold)/A(data, threshold);
		nu = (B(data)-B(data, threshold))/(A(data)-A(data, threshold));
		p = A(data, threshold)/A(data);
		q = (A(data)-A(data, threshold)) / A(data);
		sigma2 = C(data, threshold)/A(data, threshold)-(mu*mu);
		tau2 = (C(data)-C(data, threshold)) / (A(data)-A(data, threshold)) - (nu*nu);


		//The terms of the quadratic equation to be solved.
		w0 = 1.0/sigma2-1.0/tau2;
		w1 = mu/sigma2-nu/tau2;
		w2 = (mu*mu)/sigma2 - (nu*nu)/tau2 + std::log10((sigma2*(q*q))/(tau2*(p*p)));

		//If the next threshold would be imaginary, return with the current one.
		sqterm = (w1*w1)-w0*w2;
		if (sqterm < 0) {
			ucas::warning("MinError(I): not converging. Try \'Ignore black/white\' options");
			return threshold;
		}

		//The updated threshold is the integer part of the solution of the quadratic equation.
		Tprev = threshold;
		temp = (w1+std::sqrt(sqterm))/w0;

		if ( ucas::is_nan(temp)) {
			ucas::warning("MinError(I): NaN, not converging. Try \'Ignore black/white\' options");
			threshold = Tprev;
		}
		else
			threshold =(int) std::floor(temp);
	}
	return threshold;
}

// Iterative procedure based on the isodata algorithm [T.W. Ridler, S. Calvard, Picture 
// thresholding using an iterative selection method, IEEE Trans. System, Man and Cybernetics, SMC-8 (1978) 630-632.] 
int ucas::getIsoDataAutoThreshold(const std::vector<int> & data) 
{
	// IMPLEMENTATION: taken from ImageJ
	int i, l, toth, totl, h, g=0;
	for (i = 1; i < data.size(); i++){
		if (data[i] > 0){
			g = i + 1;
			break;
		}
	}
	while (true){
		l = 0;
		totl = 0;
		for (i = 0; i < g; i++) {
			totl = totl + data[i];
			l = l + (data[i] * i);
		}
		h = 0;
		toth = 0;
		for (i = g + 1; i < data.size(); i++){
			toth += data[i];
			h += (data[i]*i);
		}
		if (totl > 0 && toth > 0){
			l /= totl;
			h /= toth;
			if (g == (int) round((l + h) / 2.0))
				break;
		}
		g++;
		if (g >data.size()-2){
			ucas::warning("IsoData Threshold not found.");
			return -1;
		}
	}
	return g;
}

int ucas::getTriangleAutoThreshold(const std::vector<int> & data2) 
{   
	std::vector<int> data = data2;

	// find min and max
	int min = 0, dmax=0, max = 0, min2=0;
	for (int i = 0; i < data.size(); i++) {
		if (data[i]>0){
			min=i;
			break;
		}
	}
	if (min>0) min--; // line to the (p==0) point, not to data[min]

	// The Triangle algorithm cannot tell whether the data is skewed to one side or another.
	// This causes a problem as there are 2 possible thresholds between the max and the 2 extremes
	// of the histogram.
	// Here I propose to find out to which side of the max point the data is furthest, and use that as
	//  the other extreme.
	for (int i = int(data.size()) - 1; i >0; i-- ) {
		if (data[i]>0){
			min2=i;
			break;
		}
	}
	if (min2<data.size() - 1) min2++; // line to the (p==0) point, not to data[min]

	for (int i =0; i < data.size(); i++) {
		if (data[i] >dmax) {
			max=i;
			dmax=data[i];
		}
	}
	// find which is the furthest side
	//IJ.log(""+min+" "+max+" "+min2);
	bool inverted = false;
	if ((max-min)<(min2-max)){
		// reverse the histogram
		//IJ.log("Reversing histogram.");
		inverted = true;
		int left  = 0;          // index of leftmost element
		int right = int(data.size()) - 1; // index of rightmost element
		while (left < right) {
			// exchange the left and right elements
			int temp = data[left]; 
			data[left]  = data[right]; 
			data[right] = temp;
			// move the bounds toward the center
			left++;
			right--;
		}
		min=int(data.size()) - 1-min2;
		max=int(data.size()) - 1-max;
	}

	if (min == max){
		//IJ.log("Triangle:  min == max.");
		return min;
	}

	// describe line by nx * x + ny * y - d = 0
	double nx, ny, d;
	// nx is just the max frequency as the other point has freq=0
	nx = data[max];   //-min; // data[min]; //  lowest value bmin = (p=0)% in the image
	ny = min - max;
	d = std::sqrt(nx * nx + ny * ny);
	nx /= d;
	ny /= d;
	d = nx * min + ny * data[min];

	// find split point
	int split = min;
	double splitDistance = 0;
	for (int i = min + 1; i <= max; i++) {
		double newDistance = nx * i + ny * data[i] - d;
		if (newDistance > splitDistance) {
			split = i;
			splitDistance = newDistance;
		}
	}
	split--;

	if (inverted) {
		// The histogram might be used for something else, so let's reverse it back
		int left  = 0; 
		int right = int(data.size()) - 1;
		while (left < right) {
			int temp = data[left]; 
			data[left]  = data[right]; 
			data[right] = temp;
			left++;
			right--;
		}
		return (int(data.size()) - 1-split);
	}
	else
		return split;
}

// output(x,y) = maxval if input(x,y) > threshold, 0 otherwise
// *** WARNING *** : binarization is done in place (8-bit output)
cv::Mat ucas::binarize(cv::Mat & image, int threshold) 
{
	// checks
	if(!image.data)
		UCAS_THROW("in binarize(): invalid image");
	if(image.channels() != 1)
		UCAS_THROW("in binarize(): unsupported number of channels");
	if(image.depth() != CV_8U && image.depth() != CV_16U)
		UCAS_THROW("in binarize(): unsupported bitdepth: only 8- and 16-bit grayscale image are supported");

	// binarization
	if(image.depth() == CV_8U)
	{
		for(int i=0; i<image.rows; i++)
		{
			ucas::uint8* punt = image.ptr<ucas::uint8>(i);
			for(int j=0; j<image.cols; j++)
				punt[j] = punt[j] > threshold ? 255 : 0;
		}
	}
	else if(image.depth() == CV_16U)
	{
		for(int i=0; i<image.rows; i++)
		{
			ucas::uint16* punt = image.ptr<ucas::uint16>(i);
			for(int j=0; j<image.cols; j++)
				punt[j] = punt[j] > threshold ? 255 : 0;
		}
	}

	// conversion
	if(image.depth() != CV_8U)
		image.convertTo(image, CV_8U);

	return image;
}

//return image histogram (the number of bins is automatically set to 2^imgdepth if not provided)
std::vector<int> ucas::histogram(const cv::Mat & image, int bins /*= -1 */) 
{
	// checks
	if(!image.data)
		UCAS_THROW("in histogram(): invalid image");
	if(image.channels() != 1)
		UCAS_THROW("in histogram(): unsupported number of channels");

	// the number of gray levels
	int grayLevels  = static_cast<int>( std::pow(2, ucas::imdepth(image.depth())) );

	// computing the number of bins
	bins = bins == -1 ? grayLevels : bins;

	// input-output parameters of cv::calcHist function
	int histSize[1]  = {bins};				// number of bins
	int channels[1]  = {0};					// only 1 channel used here
	float hranges[2] = {0.0f, static_cast<float>(grayLevels)};	// [min, max) pixel levels to take into account
	const float* ranges[1] = {hranges};		// [min, max) pixel levels for all the images (here only 1 image)
	cv::MatND histo;						// where the output histogram is stored

	// histogram computation
	cv::calcHist(&image, 1, channels, cv::Mat(), histo, 1, histSize, ranges);

	// conversion from MatND to vector<int>
	std::vector<int> hist;
	for(int i=0; i<bins; i++)
		hist.push_back(static_cast<int>(histo.at<float>(i)));

	return hist;
}

// bracket the histogram to the range that holds data
std::vector<int> ucas::compressHistogram(std::vector<int> &histo, int & minbin)
{
	int maxbin=-1;
	minbin = -1;
	for (int i=0; i<histo.size(); i++)
		if (histo[i]>0) 
			maxbin = i;
	for (int i=int(histo.size())-1; i>=0; i--)
		if (histo[i]>0) 
			minbin = i;

	std::vector<int> data2(maxbin-minbin+1);
	for (int i=minbin; i<=maxbin; i++)
		data2[i-minbin]= histo[i];

	return data2;
}

// apply affine and/or warping transform to the given ROI
cv::Mat ucas::geometricTransformROI(
	cv::Mat roi,					// input ROI (should be a ROI in a parent image)
	double angle,					// rotation angle
	cv::Point2f shift,				// shift
	double scale,					// scale factor
	int *warps,						// 8 warping shifts (4 along x and 4 y) to add to each ROI point (from top-left clockwise)
	bool debug)						// debug mode: draw original/transformed roi on top parent image

{
	// get offset of roi in parent image and check roi != parent image
	cv::Size  size;
	cv::Point offset;
	roi.locateROI(size, offset);
	cv::Mat parent = roi;
	parent.adjustROI(offset.y, size.height - roi.rows, offset.x, size.width- roi.cols);


	// check roi != parent (roi is a true roi)
	if(roi.cols == parent.cols && roi.rows == parent.rows)
		UCAS_THROW("in affineTransformRoi(): ROI and parent image have the same dimensions --> ROI is not a true ROI");


	// get rotation matrix in local (roi) coordinates
	cv::Mat ROT_loc = getRotationMatrix2D(cv::Point2f(roi.cols/2.0f, roi.rows/2.0f) + shift, angle, scale);


	// transform roi in local (roi) coordinates
	cv::Point2f roi_points[4];
	roi_points[0].x = 0;
	roi_points[0].y = 0;
	roi_points[1].x = float(roi.cols);
	roi_points[1].y = 0;
	roi_points[2].x = float(roi.cols);
	roi_points[2].y = float(roi.rows);
	roi_points[3].x = 0;
	roi_points[3].y = float(roi.rows);
	cv::Mat roi_loc = (cv::Mat_<double>(3,4) << roi_points[0].x, roi_points[1].x, roi_points[2].x, roi_points[3].x,
		roi_points[0].y, roi_points[1].y, roi_points[2].y, roi_points[3].y,
		1, 1, 1, 1);
	cv::Mat roi_rot_loc = ROT_loc * roi_loc; 

	if(warps)
	{
		cv::Mat roi_loc_warped = roi_loc.clone();
		roi_loc_warped.at<double>(0,0) += warps[0];
		roi_loc_warped.at<double>(0,1) += warps[1];
		roi_loc_warped.at<double>(0,2) += warps[2];
		roi_loc_warped.at<double>(0,3) += warps[3];
		roi_loc_warped.at<double>(1,0) += warps[4];
		roi_loc_warped.at<double>(1,1) += warps[5];
		roi_loc_warped.at<double>(1,2) += warps[6];
		roi_loc_warped.at<double>(1,3) += warps[7];
		roi_rot_loc = ROT_loc * roi_loc_warped; 
	}

	// get minimum bounding rect that contains both rois in global (parent) coordinates
	cv::Rect_<double> bbox(ucas::infinity<float>(), ucas::infinity<float>(), -ucas::infinity<float>(), -ucas::infinity<float>());
	for(int i=0; i<2; i++)
	{
		double *quadrangle_loc_row = roi_loc.ptr<double>(i);
		double *quadrangle_rot_loc_row = roi_rot_loc.ptr<double>(i);
		for(int j=0; j<4; j++)
			if(i)	// y-coordinates
			{
				bbox.y = std::min(bbox.y, quadrangle_loc_row[j]);
				bbox.y = std::min(bbox.y, quadrangle_rot_loc_row[j]);
				bbox.height = std::max(bbox.height, quadrangle_loc_row[j]);
				bbox.height = std::max(bbox.height, quadrangle_rot_loc_row[j]);
			}
			else    // x-coordinates
			{
				bbox.x = std::min(bbox.x, quadrangle_loc_row[j]);
				bbox.x = std::min(bbox.x, quadrangle_rot_loc_row[j]);
				bbox.width = std::max(bbox.width, quadrangle_loc_row[j]);
				bbox.width = std::max(bbox.width, quadrangle_rot_loc_row[j]);
			}
	}
	bbox.width  = bbox.width  - bbox.x + 8;
	bbox.height = bbox.height - bbox.y + 8;
	bbox.x = bbox.x + offset.x - 4;
	bbox.y = bbox.y + offset.y - 4;
	cv::Point2f bbox_points[4];
	bbox_points[0] = cv::Point2d(bbox.x, bbox.y);
	bbox_points[1] = cv::Point2d(bbox.x + bbox.width, bbox.y);
	bbox_points[2] = cv::Point2d(bbox.x + bbox.width, bbox.y + bbox.height);
	bbox_points[3] = cv::Point2d(bbox.x, bbox.y + bbox.height);


	// check bbox is within image
	if(bbox.x < 0 || bbox.y < 0 || (bbox.x + bbox.width) > parent.cols || (bbox.y + bbox.height) > parent.rows)
		return cv::Mat();

	// get roi points in global (parent) coordinates
	roi_points[0] += cv::Point2f(float(offset.x), float(offset.y));
	roi_points[1] += cv::Point2f(float(offset.x), float(offset.y));
	roi_points[2] += cv::Point2f(float(offset.x), float(offset.y));
	roi_points[3] += cv::Point2f(float(offset.x), float(offset.y));


	// get rotated roi points in global (parent) coordinates
	cv::Point2f roi_rot_points[4];
	roi_rot_points[0].x = float(roi_rot_loc.at<double>(0,0)) + offset.x;
	roi_rot_points[0].y = float(roi_rot_loc.at<double>(1,0)) + offset.y;
	roi_rot_points[1].x = float(roi_rot_loc.at<double>(0,1)) + offset.x;
	roi_rot_points[1].y = float(roi_rot_loc.at<double>(1,1)) + offset.y;
	roi_rot_points[2].x = float(roi_rot_loc.at<double>(0,2)) + offset.x;
	roi_rot_points[2].y = float(roi_rot_loc.at<double>(1,2)) + offset.y;
	roi_rot_points[3].x = float(roi_rot_loc.at<double>(0,3)) + offset.x;
	roi_rot_points[3].y = float(roi_rot_loc.at<double>(1,3)) + offset.y;


	// transform roi and rotated-roi in bbox-coordinates
	cv::Point2f bbox_offset(float(bbox.x), float(bbox.y));
	for(int i=0; i<4; i++)
	{
		roi_points[i]     -= bbox_offset; 
		roi_rot_points[i] -= bbox_offset; 
	}

	// get inverse transform from rotated roi to roi
	cv::Mat rot_bbox = warps? cv::getPerspectiveTransform(roi_rot_points, roi_points) : cv::getAffineTransform(roi_rot_points, roi_points);

	// get bbox image
	cv::Mat bbox_img = parent(bbox).clone();

	// warp rotated roi to roi so we can get the result directly from roi
	warps ? cv::warpPerspective(bbox_img, bbox_img, rot_bbox, bbox_img.size()) : cv::warpAffine(bbox_img, bbox_img, rot_bbox, bbox_img.size());

	if(debug)
	{
		// draw roi in parent image
		cv::line(parent, roi_points[0]+bbox_offset, roi_points[1]+bbox_offset, cv::Scalar(255,0,0), 2, cv::LINE_AA);
		cv::line(parent, roi_points[1]+bbox_offset, roi_points[2]+bbox_offset, cv::Scalar(255,0,0), 2, cv::LINE_AA);
		cv::line(parent, roi_points[2]+bbox_offset, roi_points[3]+bbox_offset, cv::Scalar(255,0,0), 2, cv::LINE_AA);
		cv::line(parent, roi_points[3]+bbox_offset, roi_points[0]+bbox_offset, cv::Scalar(255,0,0), 2, cv::LINE_AA);

		// draw rotated quadrangle in parent image
		cv::line(parent, roi_rot_points[0]+bbox_offset, roi_rot_points[1]+bbox_offset, cv::Scalar(0,0,255), 2, cv::LINE_AA);
		cv::line(parent, roi_rot_points[1]+bbox_offset, roi_rot_points[2]+bbox_offset, cv::Scalar(0,0,255), 2, cv::LINE_AA);
		cv::line(parent, roi_rot_points[2]+bbox_offset, roi_rot_points[3]+bbox_offset, cv::Scalar(0,0,255), 2, cv::LINE_AA);
		cv::line(parent, roi_rot_points[3]+bbox_offset, roi_rot_points[0]+bbox_offset, cv::Scalar(0,0,255), 2, cv::LINE_AA);

		// draw bounding box in parent image
		cv::rectangle(parent, bbox, cv::Scalar(0,0,0));
	}

	// clip roi
	return bbox_img(cv::Rect(int(roi_points[0].x), int(roi_points[0].y), int(roi_points[2].x-roi_points[0].x), int(roi_points[2].y-roi_points[0].y)));
}

// write a horizontal stripe of patches
void ucas::stripewrite(const std::string & path, const std::vector <cv::Mat> & patches, bool normalize) 
{
	// check preconditions
	if(patches.empty())
		UCAS_THROW("patch vector is empty");
	cv::Size size( patches[0].cols, patches[0].rows);
	int channels = patches[0].channels();
	int depth = patches[0].depth();
	for(int i=1; i<patches.size(); i++)
	{
		if(patches[i].rows != size.height || patches[i].cols != size.width)
			UCAS_THROW(ucas::strprintf("patch #%d has size %d x %d, but first patch has size %d x %d", i, patches[i].cols, patches[i].rows, size.width, size.height));
		if(patches[i].channels() != channels)
			UCAS_THROW(ucas::strprintf("patch #%d has channels = %d, but first patch has channels %d", i, patches[i].channels(), channels));
		if(patches[i].depth() != depth)
			UCAS_THROW(ucas::strprintf("patch #%d has depth = %d, but first patch has depth %d", i, patches[i].depth(), depth));
	}

	// calculate min and max
	float min = ucas::inf<float>();
	float max = -ucas::inf<float>();
	for(int i=0; i<patches.size(); i++)
	{
		double minloc = 0, maxloc = 0;
		cv::minMaxLoc(patches[i], &minloc, &maxloc);
		min = std::min(min, float(minloc));
		max = std::max(max, float(maxloc));
	}

	// allocate stripe
	cv::Mat stripe = patches[0].clone();	// this will keep the same number of channels
	cv::resize(stripe, stripe, cv::Size(size.width * int(patches.size()), size.height));
	int stripe_depth = (channels == 1 && (depth == CV_16U || depth == CV_32F || depth == CV_64F))? CV_16U : CV_8U;
	stripe.convertTo(stripe, stripe_depth);	

	// create stripe
	for(int i=0; i<patches.size(); i++)
	{
		cv::Mat roi = stripe(cv::Rect(i*size.width, 0, size.height, size.height));
		
		if(normalize)
		{
			cv::Mat patch = patches[i].clone();
			patch.convertTo(patch, CV_32F);
			patch -= min;
			patch /= (max-min);
			patch *= (stripe_depth == CV_16U ? 65535 : 255);
			patch.convertTo(roi, stripe_depth);
		}
		else
			patches[i].convertTo(roi, stripe_depth);
	}

	// write 
	cv::imwrite(path, stripe);
}

// get histogram image
cv::Mat ucas::imhist(const cv::Mat & image) 
{
	int hist_rows = 300;			// histogram image height
	int hist_cols = 500;			// histogram image width
	float hist_occ_perc = 0.99f;	// histogram occurrence percentile, defines y-axis range

	// create histogram image, white-initialized
	cv::Mat hist_img(hist_rows, hist_cols, CV_8U, cv::Scalar(255));

	// calculate histogram
	std::vector <int> histo = ucas::histogram(image, 256);
	
	// calculate histogram visualization percentile
	std::vector <int> sorted_occurrences = histo;
	std::sort(sorted_occurrences.begin(), sorted_occurrences.end());
	float perc_viz = float(sorted_occurrences[ucas::round(hist_occ_perc*sorted_occurrences.size())]);

	// generate histogram outline
	std::vector<cv::Point> histo_contour(258);
	histo_contour[0] = cv::Point(0,hist_img.rows-1);
	for(int i=0; i<histo.size(); i++)
		histo_contour[i+1] = cv::Point(ucas::round((i/255.0)*hist_img.cols), hist_img.rows-1-ucas::round((histo[i]/perc_viz)*255));
	histo_contour.back() = cv::Point(hist_img.cols-1, hist_img.rows-1);

	// generate histogram mask
	cv::Mat mask(hist_img.size(), CV_8U, cv::Scalar(0));
	std::vector < std::vector < cv::Point> > contours;
	contours.push_back(histo_contour);
	cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);

	// generate gray shades image
	cv::Mat grayShades(hist_img.size(), CV_8U, cv::Scalar(0));
	for(int y=0; y<grayShades.rows; y++)
	{
		unsigned char* ythRow = grayShades.ptr<unsigned char>(y);
		for(int x=0; x<grayShades.cols; x++)
			ythRow[x] = static_cast<unsigned char> ( (x / float(grayShades.cols-1) ) * 255);
	}

	// copy gray shades image to histogram image under histogram outline mask
	grayShades.copyTo(hist_img, mask);

	// draw histogram outline
	cv::drawContours(hist_img, contours, -1, cv::Scalar(0), 1);

	return hist_img;
}

// get heat map of the given floating-point [0,1]-valued matrix
cv::Mat ucas::heatMap(const cv::Mat & float_mat, bool normalize, bool invert) 
{
	// check preconditions
	if(float_mat.depth() != CV_32F)
		UCAS_THROW("Unsupported bitdepth");

	cv::Mat prob = float_mat.clone();

	if(normalize)
		cv::normalize(float_mat, prob, 0, 1, cv::NORM_MINMAX);
	if(invert)
		prob = 1 - prob;

	// generate Hue heat map
	prob = (240 - prob*240.0f);

	// build hsv image
	cv::Mat _hsv[3], hsv;
	_hsv[0] = prob;
	_hsv[1] = cv::Mat::ones(prob.size(), CV_32F);
	_hsv[2] = cv::Mat::ones(prob.size(), CV_32F);
	cv::merge(_hsv, 3, hsv);

	// convert to BGR
	cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
	hsv.convertTo(hsv, CV_8U, 255);

	return hsv;
}