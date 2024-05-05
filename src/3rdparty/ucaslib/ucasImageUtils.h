#ifndef _UCAS_IMAGE_UTILS_H
#define _UCAS_IMAGE_UTILS_H

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ucasMathUtils.h"
#include "ucasExceptions.h"
#include "ucasLog.h"
#include <vector>
#include <typeinfo>
#include <iostream>

/*****************************************************************
*   OpenCV utility methods       								 *
******************************************************************/
namespace ucas
{
	// converts the OpenCV depth flag into the corresponding bitdepth
	int     imdepth(int ocv_depth);

	// detects bitdepth from maximum value of the image
	int		imdepth_detect(const cv::Mat & img);

	// extends the OpenCV "imshow" function with the addition of a scale factor
	void    imshow(const std::string winname, cv::InputArray arr, bool wait = true, float scale = 1.0);

	// get histogram image
	cv::Mat imhist(const cv::Mat & image) ;

	// extends the OpenCV "imread" function by adding support to OpenCV mat (.cvmat) and grayscale 8-16 bits DICOM images (.dcm) 
	// please note that for DICOM images:
	// - 'opencv_flags' are ignored
	// - pixel values are NOT changed
	// - 'bits_used' returns the bits used 
	cv::Mat imread(const std::string & path, int opencv_flags = 1, int *bits_used = 0) ;

	// extends the OpenCV "imwrite" function by adding support to OpenCV mat (.cvmat) and grayscale 8-16 bits DICOM images (.dcm)
	void    imwrite(const std::string & path, cv::Mat & img) ;

	// write a horizontal stripe of patches
	void    stripewrite(const std::string & path, const std::vector <cv::Mat> & patches, bool normalize = false) ;

	// rescale the image from 'bits_in' to 'bits_out'
	void    imrescale(cv::Mat & image, int bits_in, int bits_out) ;

	// return (x,y) offset of 'data' in parent image (if any), otherwise return (0,0)
	cv::Point imOffsetInParent(const cv::Mat & img);

	// get heat map of the given floating-point [0,1]-valued matrix
	cv::Mat heatMap(const cv::Mat & float_mat, bool normalize, bool invert) ;
}

/*****************************************************************
*   Image binarization methods   								 *
******************************************************************/
namespace ucas
{
	enum binarizationMethod
	{
		all,											// try all implemented binarization techniques
		otsuopencv,										// Otsu binarization (opencv)
		otsu,											// Otsu binarization 
		isodata,										// see [T.W. Ridler, S. Calvard, "Picture thresholding using an iterative selection method", IEEE Trans. System, Man and Cybernetics, SMC-8 (1978) 630-632]
		triangle,										// see [Zack GW, Rogers WE, Latt SA (1977), "Automatic measurement of sister chromatid exchange frequency", J. Histochem. Cytochem. 25 (7): 741–53]
		mean,											// see [C. A. Glasbey, "An analysis of histogram-based thresholding algorithms," CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993]
		minerror,										// see [ Kittler and J. Illingworth, "Minimum error thresholding," Pattern Recognition, vol. 19, pp. 41-47, 1986]
		maxentropy,										// see [ Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy of the Histogram" Graphical Models and Image Processing, 29(3): 273-285]
		renyientropy,									// see [ Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy of the Histogram" Graphical Models and Image Processing, 29(3): 273-285]
		yen												// see [ Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion for Automatic Multilevel Thresholding" IEEE Trans. on Image Processing, 4(3): 370-378]
	};
	std::string binarizationMethod_toString(binarizationMethod code);
	binarizationMethod binarizationMethod_toInt(const std::string & code);
	std::string binarizationMethods();

	// C. A. Glasbey, "An analysis of histogram-based thresholding algorithms,"
	// CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.
	int getMeanThreshold(const std::vector<int> & data) ;

	// Kittler and J. Illingworth, "Minimum error thresholding," Pattern Recognition, vol. 19, pp. 41-47, 1986.
	// C. A. Glasbey, "An analysis of histogram-based thresholding algorithms," CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.
	int getMinErrorIThreshold(const std::vector<int> & data) ;

	// Iterative procedure based on the isodata algorithm [T.W. Ridler, S. Calvard, Picture 
	// thresholding using an iterative selection method, IEEE Trans. System, Man and Cybernetics, SMC-8 (1978) 630-632.] 
	int getIsoDataAutoThreshold(const std::vector<int> & data) ;

	// Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
	// Automatic Measurement of Sister Chromatid Exchange Frequency, Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753
	int getTriangleAutoThreshold(const std::vector<int> & data) ;

	// Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) 
	// "A New Method for Gray-Level Picture Thresholding Using the Entropy of the Histogram" Graphical Models and Image Processing, 29(3): 273-285
	int getMaxEntropyAutoThreshold(const std::vector<int> & data) ;

	// Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) 
	// "A New Method for Gray-Level Picture Thresholding Using the Entropy of the Histogram" Graphical Models and Image Processing, 29(3): 273-285
	int getRenyiEntropyAutoThreshold(const std::vector<int> & data) ;

	//Yen J.C., Chang F.J., and Chang S. (1995) 
	// "A New Criterion for Automatic Multilevel Thresholding" IEEE Trans. on Image Processing, 4(3): 370-378
	int getYenyAutoThreshold(const std::vector<int> & data) ;

	// Otsu's threshold algorithm
	int getOtsuAutoThreshold(const std::vector<int> & data) ;

	// output(x,y) = maxval if input(x,y) > threshold, 0 otherwise
	// *** WARNING *** : binarization is done in place (8-bit output)
	cv::Mat binarize(cv::Mat & image, int threshold) ;
	/*-------------------------------------------------------------------------------------------------------------------------*/
}

/*****************************************************************
*   Image processing functions									 *
******************************************************************/
namespace ucas
{
	// return image histogram (the number of bins is automatically set to 2^imgdepth if not provided)
	std::vector<int> histogram(const cv::Mat & image, int bins = -1 ) ;
	
	// bracket the histogram to the range that holds data
	std::vector<int> compressHistogram(std::vector<int> &histo, int & minbin);

	// apply affine and/or warping transform to the given ROI
	cv::Mat geometricTransformROI(
		cv::Mat roi,									// input ROI (should be a ROI in a parent image)
		double angle=0,									// rotation angle
		cv::Point2f shift = cv::Point2f(0.0, 0.0),		// shift
		double scale = 1.0,								// scale factor
		int *warps = 0,									// 8 warping shifts (4 along x and 4 y) to add to each ROI point (from top-left clockwise)
		bool debug = false)								// debug mode: draw original/transformed roi on top parent image
	;
}

/*****************************************************************
*   Data structures              								 *
******************************************************************/
namespace ucas
{
	struct detected_window
	{
		int x,y, height, width;
		float detector_score; 
		cv::Rect rect;
		bool true_positive;

		detected_window(int _x, int _y, int _win_width, int _win_height, float _conf_degree) : x(_x), y(_y), height(_win_height), width(_win_width), detector_score(_conf_degree)
		{
			rect.x = x;
			rect.y = y;
			rect.width = width;
			rect.height = height;
			true_positive = false;
		}
		detected_window(int _x, int _y, float _conf_degree) : x(_x), y(_y), detector_score(_conf_degree){height=width=-1;true_positive=false;}
		detected_window(){x=y=0; detector_score=0.0F; height=width=-1;true_positive=false;}
	};
}


/*****************************************************************
*   N-dimensional blob methods   					             *
******************************************************************/
namespace ucas
{
	template <typename T>
	class Blob
	{
		private:

			T* _data;
			std::vector<size_t> _shape;
			std::string _type;
			size_t _offset;					// with respect to first dimension

		public:

			// default constructor
			Blob() : _data(0), _type(typeid(T).name()), _offset(0) {}

			// constructor 1: both data and shape are provided
			Blob(T* data, std::vector<size_t> & shape) 
			{
				_type = typeid(T).name();
				_shape = shape;
				_data = data;
				_offset = 0;
			}

			// constructor 2: only shape is provided, data are allocated and initialized with the given value
			Blob(std::vector<size_t> & shape, bool allocate_memory = true, T init_val = 0)  
			{
				_type = typeid(T).name();
				_shape = shape;
				_data = 0;
				_offset = 0;

				if(allocate_memory)
				{
					try
					{
						_data = new T[size()];
					}
					catch (std::bad_alloc & e)
					{
						throw Error(strprintf("Failed to allocate %.2f Gigabytes for blob data", double(size())/1000000000.0));
					}
					for(size_t i=0; i<size(); i++)
						_data[i] = init_val;
				}
			}

			// destructor
			~Blob(){}

			// release
			void release(){if(_data) delete[] _data; _data = 0; _shape.clear();}

			// getters
			T* data() const {if(!_data) return 0; else return &(_data[_offset*size(true)]);}
			T* data_no_offset() const{return _data;}
			bool empty(){return _data == 0;}
			std::vector<size_t> shape() const {return _shape;}
			std::string type() const {return _type;}
			size_t offset() const {return _offset;}
			size_t size(bool exclude_first_dimension = false) const
			{
				size_t d = 1;
				for(size_t i = exclude_first_dimension ? 1 : 0; i<_shape.size(); i++)
					d *= _shape[i];
				return d;
			}

			// setters
			void setData(T* data){_data = data;}
			void setShape(size_t at, size_t val) 
			{
				if(at<_shape.size())
					_shape[at] = val;
				else
					throw Error(strprintf("Out of range dimension in setShape(%d) [_shape.size() = %d]", at, _shape.size()));
			}
			void setOffset(size_t new_offset){ _offset = new_offset; }
			void resetOffset(){_offset = 0;}

			// print
			void print(bool all_values = false)
			{
				printf("Dimensions: ");
				for(size_t i=0; i<_shape.size(); i++)
					printf("%d %s", _shape[i], i == _shape.size() - 1 ? "\n" : "x ");
				printf("Type: \"%s\"\n", _type.c_str());
				printf("Data: ");
				if(size() >= 3 && !all_values)
					std::cout << _data[0] << " " << _data[1] << " " << _data[2];
				else if(all_values)
					for(size_t i=0; i<size(); i++)
						std::cout << _data[i] << " ";
				T avg = 0;
				for(size_t i=0; i<size(); i++)
					avg += _data[i];
				std::cout << " ... average = " << avg/size() << std::endl;
			}

			// calculate average of samples
			double average(
				interval<size_t> inclusion_range =	// inclusion range selection applied on the first dimension/axis
				interval<size_t>(0, inf<size_t>()),	// DEFAULT = all
				interval<size_t> exclusion_range =	// inclusion range selection applied on the first dimension/axis
				interval<size_t>(0, 0)				// DEFAULT = none
				)
			{
				double avg = 0;
				double count = 0;
				size_t one_blob_dim = size(true);
				for(size_t i=0; i<_shape[0]; i++)
					if( (i >= inclusion_range.start && i < inclusion_range.end) &&
					   !(i >= exclusion_range.start && i < exclusion_range.end))
					{
						for(size_t j=0; j<one_blob_dim; j++)
						{
						   avg += double(_data[i*one_blob_dim+j]);
						   count++;
						}
					}
				return avg/count;
			}

	};

	template <typename T>
	void	blobwrite(
		const std::string path,				// absolute file path
		const Blob<T> & blob,				// blob
		const std::string & header = ""		// header (optional)
	) 
	{
		// open file
		FILE* f = fopen(path.c_str(), "wb");
		if(!f)
			throw CannotWriteFileError(path);
		
		// write header
		size_t header_size = header.size()+1;
		if(!fwrite(&header_size, sizeof(size_t), 1, f))
			throw Error(strprintf("Cannot write header size in file \"%s\"", path.c_str()));
		if(fwrite(header.c_str(), sizeof(char), header_size, f) != header_size)
			throw Error(strprintf("Cannot write header string in file \"%s\"", path.c_str()));

		// write type name
		std::string type_name = typeid(T).name();
		size_t type_name_size = type_name.size()+1;
		if(!fwrite(&type_name_size, sizeof(size_t), 1, f))
			throw Error(strprintf("Cannot write type name size in file \"%s\"", path.c_str()));
		if(fwrite(type_name.c_str(), sizeof(char), type_name_size, f) != type_name_size)
			throw Error(strprintf("Cannot write type name string in file \"%s\"", path.c_str()));
		
		// write blob shape dimensions
		size_t blob_shape_size = blob.shape().size();
		if(!fwrite(&blob_shape_size, sizeof(size_t), 1, f))
			throw Error(strprintf("Cannot write blob shape size in file \"%s\"", path.c_str()));
		for(auto & dim_i : blob.shape())
			if(!fwrite(&dim_i, sizeof(size_t), 1, f))
				throw Error(strprintf("Cannot write blob shape dim in file \"%s\"", path.c_str()));
		
		// write blob data
		if(fwrite(blob.data(), sizeof(T), blob.size(), f) != blob.size())
			throw Error(strprintf("Cannot write blob data in file \"%s\"", path.c_str()));

		// close file
		fclose(f);
	}

	template <typename T>
	Blob<T>	blobread(
		const std::string path,				// absolute file path
		interval<size_t> in_range =	0,		// inclusion range selection applied on the first dimension/axis
		interval<size_t> ex_range =	0,		// exclusion range selection applied on the first dimension/axis
		std::string * header = 0,			// header (optional)
		bool read_only_metadata = false,	// whether to read only metadata
		Blob<T> * blob_append = 0			// append to existing blob
		) 
	{
		// reset header
		if(header)
			header->clear();

		// open file
		FILE* f = fopen(path.c_str(), "rb");
		if(!f)
			throw CannotWriteFileError(path);

		// read header
		size_t header_size = 0;
		if(!fread(&header_size, sizeof(size_t), 1, f))
			throw Error(strprintf("Cannot read header size in file \"%s\"", path.c_str()));
		char * header_cstr = new char[header_size];
		if(fread(header_cstr, sizeof(char), header_size, f) != header_size)
			throw Error(strprintf("Cannot read header string in file \"%s\"", path.c_str()));
		if(header)
			*header = header_cstr;
		delete[] header_cstr;

		// read type name
		size_t type_name_size = 0;
		if(!fread(&type_name_size, sizeof(size_t), 1, f))
			throw Error(strprintf("Cannot read type name size in file \"%s\"", path.c_str()));
		char * type_name_cstr = new char[type_name_size];
		if(fread(type_name_cstr, sizeof(char), type_name_size, f) != type_name_size)
			throw Error(strprintf("Cannot read type name string in file \"%s\"", path.c_str()));
		//if(std::string(typeid(T).name()) != std::string(type_name_cstr))
			//warning(strprintf("Data type mismatch in file \"%s\": expected \"%s\", found \"%s\"", path.c_str(), typeid(T).name(), type_name_cstr));
		delete[] type_name_cstr;

		// read blob shape dimensions
		size_t blob_shape_size = 0;
		if(!fread(&blob_shape_size, sizeof(size_t), 1, f))
			throw Error(strprintf("Cannot read blob shape size in file \"%s\"", path.c_str()));
		if(!blob_shape_size)
			throw Error(strprintf("Invalid (=0) blob shape size in file \"%s\"", path.c_str()));
		std::vector<size_t> blob_shape(blob_shape_size);
		for(auto & dim_i : blob_shape)
			if(!fread(&dim_i, sizeof(size_t), 1, f))
				throw Error(strprintf("Cannot read blob shape dim in file \"%s\"", path.c_str()));

		// apply range selections
		if(in_range.end == 0)
			in_range = blob_shape[0];
		if(!in_range.isValid())
			throw Error(strprintf("Invalid inclusion range [%d,%d) in file \"%s\"", in_range.start, in_range.end, path.c_str()));
		if(in_range.start >= blob_shape[0])
			throw Error(strprintf("Invalid inclusion range [%d,%d) in file \"%s\" that has shape[0]=%d", in_range.start, in_range.end, path.c_str(), blob_shape[0]));
		if(ex_range.start > ex_range.end)
			throw Error(strprintf("Invalid exclusion range [%d,%d) in file \"%s\"", ex_range.start, ex_range.end, path.c_str()));
		auto ranges = in_range.subtract(ex_range);
		if(ranges.first.size() == 0 && ranges.second.size() == 0)
			throw Error(strprintf("Empty range selection in file \"%s\" that has shape[0]=%d resulting from inclusion [%d,%d) and exclusion [%d,%d) ranges", path.c_str(), blob_shape[0], in_range.start, in_range.end, ex_range.start, ex_range.end));
		blob_shape[0] = ranges.first.size() + ranges.second.size();

		// allocate memory if needed
		Blob<T> blob(blob_shape, !read_only_metadata && !blob_append);

		// check if blob append is possible and set append point
		if(blob_append)
		{
			if(std::string(typeid(T).name()) != blob_append->type())
				throw Error(strprintf("Cannot append blob in file \"%s\" to existing blob: expected type \"%s\", found \"%s\"", path.c_str(), blob_append->type().c_str(), typeid(T).name()));
			if(blob_shape.size() != blob_append->shape().size())
				throw Error(strprintf("Cannot append blob in file \"%s\" to existing blob: shape size mismatch", path.c_str()));
			for(size_t i=1; i<blob_shape.size(); i++)
				if(blob_shape[i] != blob_append->shape()[i])
					throw Error(strprintf("Cannot append blob in file \"%s\" to existing blob: shape dimension mismatch (%d != %d)", path.c_str(), blob_shape[i], blob_append->shape()[i]));
			if(blob_append->offset() + blob.shape()[0] > blob_append->shape()[0])
				throw Error(strprintf("Cannot append blob in file \"%s\" to existing blob: not enough space (%d + %d > %d)", path.c_str(), blob_append->offset(), blob.shape()[0], blob_append->shape()[0]));

			blob.setData(blob_append->data());	// set append point based on blob_append's internal offset
		}

		// read blob data
		if(!read_only_metadata)
		{
			size_t start1 = ranges.first.start*blob.size(true)*sizeof(T);
			size_t count1 = ranges.first.size()*blob.size(true);
			if(fseek(f, start1, SEEK_CUR))
				throw Error(strprintf("Cannot seek blob data in file \"%s\" at position %d", path.c_str(), start1));
			if(fread(blob.data(), sizeof(T), count1, f) != count1)
				throw Error(strprintf("Cannot read blob data in file \"%s\"", path.c_str()));

			if(ranges.second.size())
			{
				size_t start2 = ranges.second.start*blob.size(true)*sizeof(T) - (start1+count1*sizeof(T));
				size_t count2 = ranges.second.size()*blob.size(true);
				if(fseek(f, start2, SEEK_CUR))
					throw Error(strprintf("Cannot seek blob data in file \"%s\" at position %d", path.c_str(), start1));
				if(fread(blob.data() + count1, sizeof(T), count2, f) != count2)
					throw Error(strprintf("Cannot read blob data in file \"%s\"", path.c_str()));
			}
		}

		// close file
		fclose(f);

		// return blob
		return blob_append ? *blob_append : blob;
	}
}

#endif