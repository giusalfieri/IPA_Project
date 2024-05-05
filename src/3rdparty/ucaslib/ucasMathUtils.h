#ifndef _UCAS_MATH_UTILS_H
#define _UCAS_MATH_UTILS_H

#include <limits>
#include <vector>
#include <fstream>
#include "google_almostequals.hpp"
#include "ucasExceptions.h"
#include "ucasStringUtils.h"
#include <opencv2/core/core.hpp>

namespace ucas
{
	/*******************
    *    CONSTANTS     *
    ********************
    ---------------------------------------------------------------------------------------------------------------------------*/
	const double PI = 3.14159265;				//pi
	const double LOG2E = 1.44269504088896340736; //log2(e)
	const float		inf_pos_r32		=  std::numeric_limits<float>::infinity();
	const float		inf_neg_r32		= -std::numeric_limits<float>::infinity();
	const double	inf_pos_r64		=  std::numeric_limits<double>::infinity();
	const double	inf_neg_r64		= -std::numeric_limits<double>::infinity();
	const int		inf_pos_si32	=  std::numeric_limits<int>::max();
	const int		inf_neg_si32	=  std::numeric_limits<int>::min();
    /*-------------------------------------------------------------------------------------------------------------------------*/

	// floating point comparisons
	// The following definitions are from The art of computer programming by Knuth
	template <typename T>
	bool approximatelyEqual(T a, T b, T epsilon)
	{
		return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
	}
	template <typename T>
	bool essentiallyEqual(T a, T b, T epsilon)
	{
		return fabs(a - b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
	}
	// The following functions are taken from Google
	template <typename T>
	inline bool areEqual(const T & a, const T & b){
		const FloatingPoint<T> lhs(a), rhs(b);
		return lhs.AlmostEquals(rhs);
	}
	template <typename T>
	inline bool areNotEqual(const T & a, const T & b){
		return !areEqual<T>(a,b);
	}

	// interval [start, end)
	template <typename T = int>
	struct interval
	{
		T start, end;
		interval() : start(0), end(0){}
		interval(T _start, T _end) : start(_start), end(_end){}
		interval(T _dims) : start(0), end(_dims){}
		T size(){return T(fabs(double(start)-end));}
		std::pair< interval<T>, interval<T> > subtract(interval<T> i)
		{
			std::pair< interval<T>, interval<T> > result;

			if(i.start > start || i.end < start)
				result.first.start = start;
			if(i.start <= start && i.end >= start && i.end < end)
				result.first.start = i.end;
			if(i.start >= end || (i.start <= start && i.end < end))
				result.first.end = end;
			if(i.start > start && i.start < end)
				result.first.end = i.start;
			if(i.start > start && i.end < end)
			{
				result.second.start = i.end;
				result.second.end = end;
			}

			return result;
		}
		bool contains(int i){return i >= start && i<end;}
		bool isValid(){return end > start;}
	};
	typedef ucas::interval<int> range;

	// distance function
	template <class T>
	inline T distance(T& x1, T& y1, T& x2, T& y2){
		return static_cast<T>( sqrt( static_cast<double>(x1 - x2) * (x1 - x2) + static_cast<double>(y1 - y2) * (y1 - y2)) );
	}

	// log2 function
	template <typename T>
	inline T log2(const T& x){
		return static_cast<T>(std::log(x) * LOG2E);
	}

	// round function
	template <class T> 
	inline int round(T x) {
		return static_cast<int>(x > 0 ? x + 0.5 : x - 0.5);
	}

	// abs function
	template <class T>
	inline T abs(const T & x){
		if(x < 0)
			return -x;
		else
			return x;
	}

	// partition 'elems' into 'nParts' which differ at most by 1
	inline std::vector < interval<int> > partition(interval<int> elems, size_t nParts)
	{
		std::vector < interval<int> > parts(nParts);
		size_t n_elems_i = elems.size()/nParts;
		size_t elem_count = elems.start;
		for(int i=0; i<parts.size(); i++)
		{
			parts[i].start	= int(elem_count);
			parts[i].end	= int(elem_count + n_elems_i + (i < elems.size()%nParts ? 1 : 0));
			elem_count += parts[i].size();
		}
		return parts;
	}

	template <typename T>
	T infinity(){ return std::numeric_limits<T>::infinity();}

	template <typename T>
	T ssqrt(T x){ return x <= static_cast<T>(0) ? static_cast<T>(0) : sqrt(x);}

	template <typename T>
	T inf(){ 
		if(std::numeric_limits<T>::has_infinity)
			return std::numeric_limits<T>::infinity();
		else
			return std::numeric_limits<T>::max();
	}

	template<typename T> T ninf(){ 
		if(std::numeric_limits<T>::has_infinity)
			return -std::numeric_limits<T>::infinity();
		else
			return std::numeric_limits<T>::min();
	}

	// isnan
#include <cmath>
#ifdef _WIN32
	inline bool is_nan(double x){
		return _isnan(x) == 1;
	}
#else
	inline bool is_nan(double x){
		return std::isnan(x);
	}
#endif


	// generates all octaves between a (>=0) and b (>=0) in a logarithmic scale with base 10
	template<typename T>
	std::vector<T> octspace10(T a, T b) 
	{
		if(a <= 0 || b <= 0)
			throw "in octspace10(): either a or b are not greater than zero";
		T right = pow(10, ceil(log(b)/log(10)));
		T left = pow(10, floor(log(a)/log(10)));
		std::vector<T> result;
		for(T elem = left; elem <= right; elem*=10)
			for(int i=1; i<10; i++)
				if(elem*i >= a && elem*i <= b)
					result.push_back(elem*i);
		return result;
	}

	// divide the given range into logarithmic decades
	inline std::vector <double> decades(double a, double b)
	{
		std::vector< double> out;
		out.push_back(a);
		for(int i=int(std::floor(log10(a))); i<int(std::ceil(log10(b))); i++)
			for(int j=1; j<10; j++)
				if(j*std::pow(10.0, i) < b && j*std::pow(10.0, i) > a)
					out.push_back(j*std::pow(10.0, i));
		out.push_back(b);
		return out;
	}

	// divide the given intervals in 'n' subintervals
	inline std::vector<double> subdivide(const std::vector<double> intervals, int n)
	{
		std::vector<double> out;
		for(size_t i=0; i<intervals.size()-1; i++)
		{
			double step = (intervals[i+1]-intervals[i])/n;
			for(int j=0; j<n; j++)
				out.push_back(intervals[i] + j*step);
		}
		out.push_back(intervals.back());
		return out;
	}

	// isfinite
	template <typename T>
	inline bool isfinite(T x){
		return 
			!is_nan(x) && 
			x != std::numeric_limits<T>::infinity() &&
			x != -std::numeric_limits<T>::infinity();
	}

	template <typename T>
	inline void meanstd(T data[], size_t dim, double & mean, double & std){
		mean = std = 0;
		if(dim == 0)
			return;
		double sum=0, sumsq=0;
		for(size_t i=0; i<dim; i++)
		{
			sum += data[i];
			sumsq += data[i]*data[i];
		}
		mean = sum / static_cast<double>(dim);
		std = std::sqrt(sumsq / static_cast<double>(dim) - mean*mean);
	}

	template <typename T>
	inline void minmax(T data[], size_t dim, T & min, T & max){
		if(dim == 0)
			return;
		min = max = data[0];
		for(size_t i=1; i<dim; i++)
		{
			min = data[i] < min ? data[i] : min;
			max = data[i] > max ? data[i] : max;
		}
	}

	struct interval_t
	{
		int start, end;			//interval extremes (integer field)
		double f_start, f_end;	//interval extremes (floating point field)
		int size;				//size of interval (not necessary end-start, as in the case of histogram bins)
		double m;				//mean (weighted or not) value, only for histogram bins
		interval_t(void) : start(-1), end(-1), f_start(-1.0f), f_end(-1.0f), size(-1), m(-1)  {}
		interval_t(int _start, int _end, int _size) : start(_start), end(_end), f_start(-1.0f), f_end(-1.0f),	size(_size), m(-1){}
		interval_t(int _start, int _end) : start(_start), end(_end), f_start(-1.0f), f_end(-1.0f),	size(-1), m(-1){}
	};

	// calculate the P-percentile from the given histogram
	template <typename T>
	inline size_t prctile(
		T hist[],				// histogram
		size_t dim,				// number of histogram bins
		double P)				// percentage in the range [0, 100]
	
	{
		// check inputs
		if(dim < 1)
			throw "prctile(): dim < 1";
		if(P < 0 || P > 100)
			throw "prctile(): P not in the range [0, 100]";
		P /= 100;

		// calculate sum of all histogram values
		double sum=0;
		for(size_t i=0; i<dim; i++)
			sum += static_cast<double>(hist[i]);

		// calculate percentile
		double acc=0;
		size_t idx = 0;
		while(idx < (dim-1) && acc < P*sum)
			acc += hist[idx++];
		return idx;
	}

	inline static double str2f(const char* str)
	{
		if(stristr(str, "1.#inf")==str)
			return std::numeric_limits<double>::infinity();
		else if(stristr(str, "-1.#inf")==str)
			return -std::numeric_limits<double>::infinity();
		else if(stristr(str, "-inf")==str) 
			return -std::numeric_limits<double>::infinity();
		else if(stristr(str, "inf")==str) 
			return std::numeric_limits<double>::infinity();
		else
			return atof(str);
	}

	inline static std::string f2str(double d)
	{
		std::string result;
		if(is_nan(d))
			result = "nan";
		else if(d == inf<double>())
			result = "inf";
		else if(d == -inf<double>())
			result = "-inf";
		else
			result = num2str<double>(d);
		return result;
	}

	// calculate and save probability-density functions for the given OpenCV-data
	inline static void 
		saveDataPDF(
		std::vector <cv::Mat> & data, 
		const std::string & path, 
		int bins = 10000) 
	
	{
		// checks
		if(bins < 1)
			throw Error("in saveDataPDF(): number of bins is < 1");
		if(data.empty())
			throw Error("in saveDataPDF(): empty data");
		if(path.empty())
			throw Error("in saveDataPDF(): empty path");
		if(data[0].depth() != CV_32F)
			throw Error("in saveDataPDF(): unsupported depth, only 32-floating point images are supported");
		int channels = data[0].channels();

		// calculate max and min
		double max = -ucas::infinity<double>(), min = ucas::infinity<double>();
		for(auto & mat : data)
		{
			double lmin = 0, lmax = 0;
			cv::minMaxLoc(mat, &lmin, &lmax);
			max = std::max(max, lmax);
			min = std::min(min, lmin);
		}

		// calculate PDF
		double step = (max - min) / bins;
		std::vector<double> hist(bins, 0);
		int count = 0;
		for(auto & mat : data)
		{
			for(int y=0; y<mat.rows; y++)
			{
				float* data_row = mat.ptr<float>(y);
				for(int x=0; x<mat.cols; x++)
					for(int c=0; c<channels; c++)
					{
						count++;

						if(ucas::areEqual<float>(data_row[channels*x+c], float(max)))
							hist[bins-1]++;
						else 
						{
							int bini = static_cast<int>(  floor( (data_row[channels*x+c]-min)/static_cast<double>(step) )  );
							if(bini >= bins)			// because of limited real number precision, it might occur (in extreme cases) that bini >= bins;
								bini = bins-1;			// to prevent this, bini computation is done using double precision
							hist[bini]++;
						}
					}
			}
		}
		for(auto & h : hist)
			h /= count;

		// save file
		std::ofstream f(path);
		if(!f.is_open())
			throw CannotWriteFileError(path);
		f << "bin_mean\tprobability\n";
		for(size_t b=0; b<hist.size()-1; b++)
			f << ((min + b*step) + (min + (b+1)*step))/2 << " " << hist[b] << std::endl;
		f << max << " " << hist[hist.size()-1];
		f.close();
	}
}

namespace Maths
{
	namespace Interpolation
	{
		//! Linearly interpolates a given set of points.
		class Linear
		{
		public:

			//! Class constructor
			Linear(int _n, double *x, double *y)
			{
				n = _n;
				m_x = new double[static_cast<int>(n)];
				m_y = new double[static_cast<int>(n)];

				for (int i = 0; i < n; ++i) 
				{
					m_x[i] = x[i];
					m_y[i] = y[i];
				}
			}

			//! Class destructor
			~Linear()
			{
				delete [] m_x;
				delete [] m_y;
			}

			//! Returns an interpolated value.
			double getValue(double x)
			{
				int i = 0;
				while ( (i < n-1) && x > m_x[++i]);
				//printf("x = %.5f, i = %d, m_x[%d] = %.5f, m_x[%d] = %.5f, ", x, i, i-1, m_x[i-1], i, m_x[i]);


				double a = (x - m_x[i - 1]) / (m_x[i] - m_x[i - 1]);
				//printf("a = %.5f, m_y[%d] = %.5f, m_y[%d] = %.5f\n", a, i-1, m_y[i-1], i, m_y[i]);
				return m_y[i - 1] + a * (m_y[i] - m_y[i - 1]);
			}

		private:

			double *m_x, *m_y, n;
		};


		//! A static function implementing the Linear Class for one off calculations
		inline double Linear_once(int N, double *x, double *y, double a )
		{
			// This function is created to enable an Instant Calculator on CodeCogs. 
			// You probably shouldn't be using this function otherwise. 

			Maths::Interpolation:: Linear A(N, x, y);
			return A.getValue(a);
		}
	}
}


#endif