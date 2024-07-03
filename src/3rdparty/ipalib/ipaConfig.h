#ifndef _ipa_utils_h
#define _ipa_utils_h

#include <string>
#include <cstdarg>
#include <vector>
#include <sstream>
#include <limits>
#include <cstring>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#ifdef _WIN32
#include <ctime>
#include <direct.h>
#else
#include <time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#endif

// undefine dangerous macros if any
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif 

/*******************************************************************************************************************************
 *   Forward declarations, types, parameters, constants, exceptions and utility functions   										       *
 *******************************************************************************************************************************/
namespace ipa
{
    /*******************
    *   DECLARATIONS   *
    ********************
    ---------------------------------------------------------------------------------------------------------------------------*/
	class error;								//failure exception thrown by functions in the this module
	enum  failure_type { UNDEFINED };
	enum  debug_level { NO_DEBUG, LEV1, LEV2, LEV3 };
	enum  axis {INVALID_AXIS, X, Y, Z};
    /*-------------------------------------------------------------------------------------------------------------------------*/


    /*******************
    *       TYPES      *
    ********************
    ---------------------------------------------------------------------------------------------------------------------------*/
	typedef signed char	sint8;					//8-bit  signed   integers (-128                       -> +127)
	typedef short sint16;						//16-bit signed   integers (-32,768                    -> +32,767)
	typedef int sint32;							//32-bit signed   integers (-2,147,483,648             -> +2,147,483,647)
	typedef long long sint64;					//64-bit signed   integers (�9,223,372,036,854,775,808 -> +9,223,372,036,854,775,807)
	typedef unsigned char uint8;				//8-bit  unsigned integers (0 -> +255)
	typedef unsigned short int uint16;			//16-bit unsigned integers (0 -> +65,535)
	typedef unsigned int uint32;				//32-bit unsigned integers (0 -> +4,294,967,295)
	typedef unsigned long long uint64;			//64-bit unsigned integers (0 -> +18,446,744,073,709,551,615
	typedef float real32;						//real single precision
	typedef double real64;						//real double precision
    /*-------------------------------------------------------------------------------------------------------------------------*/


	/*******************
    *    CONSTANTS     *
    ********************
    ---------------------------------------------------------------------------------------------------------------------------*/
	const double PI = 3.14159265358979323846;	//pi
    /*-------------------------------------------------------------------------------------------------------------------------*/


  	/*******************
	*    PARAMETERS    *
	********************
    ---------------------------------------------------------------------------------------------------------------------------*/
	extern int DEBUG;							//debug level of current module
	extern int SCREEN_WIDTH;					//screen width (in pixels)
	extern int SCREEN_HEIGHT;					//screen height (in pixels)
	extern double CAMERA_FPS;					//camera (webcam) frames per second
	extern std::string FACE_DETECTOR_PATH;		//face detector xml path (haarcascade_frontalface_alt.xml)
    /*-------------------------------------------------------------------------------------------------------------------------*/


	/********************************************
	 *   Cross-platform UTILITY functions	    *
	 ********************************************
	---------------------------------------------------------------------------------------------------------------------------*/
	//infinity 
	template<class T> T inf(){ 
		if(std::numeric_limits<T>::has_infinity)
			return std::numeric_limits<T>::infinity();
		else
			return std::numeric_limits<T>::max();
	}
	template<class T> T ninf(){ 
		if(std::numeric_limits<T>::has_infinity)
			return -std::numeric_limits<T>::infinity();
		else
			return std::numeric_limits<T>::min();
	}

	//the case insensitive version of C strstr() function
	inline const char* stristr(const char *str1, const char *str2){
	   if ( !*str2 )
		  return str1;
	   for ( ; *str1; ++str1 ){
		  if ( toupper(*str1) == toupper(*str2) ){
			 const char *h, *n;
			 for ( h = str1, n = str2; *h && *n; ++h, ++n ){
				if ( toupper(*h) != toupper(*n) ){
				   break;
				}
			 }
			 if ( !*n ) /* matched all of 'str2' to null termination */
				return str1; /* return the start of the match */
		  }
	   }
	   return 0;
	}

	//the case insensitive version of C strcmp() function
	inline int stricmp (const char *s1, const char *s2){
		if (s1 == 0) return s2 == 0 ? 0 : -(*s2);
		if (s2 == 0) return *s1;

		char c1, c2;
		while ((c1 = tolower (*s1)) == (c2 = tolower (*s2)))
		{
			if (*s1 == '\0') break;
			++s1; ++s2;
		}

		return c1 - c2;
	}

	//stringstream-based integer-to-string conversion
	inline std::string int2str(const int& val){
		std::stringstream ss;
		ss << val;
		return ss.str();
	}

	//infinity-compliant string-to-double conversion
	inline double str2f(const char* str){
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

	//fgetstr() - mimics behavior of fgets(), but removes new-line character at end of line if it exists
	inline char *fgetstr(char *string, int n, FILE *stream){
		char *result;
		result = fgets(string, n, stream);
		if(!result)
			return(result);

		char *nl = strrchr(string, '\r');
		if (nl) *nl = '\0';
		nl = strrchr(string, '\n');
		if (nl) *nl = '\0';

		return(string);
	}

	//string-based tokenization function
	inline void	split(std::string& theString, std::string delim, std::vector<std::string>& tokens){
		size_t  start = 0, end = 0;
		while ( end != std::string::npos)
		{
			end = theString.find( delim, start);

			// If at end, use length=maxLength.  Else use length=end-start.
			tokens.push_back( theString.substr( start,
				(end == std::string::npos) ? std::string::npos : end - start));

			// If at end, use start=maxSize.  Else use start=end+delimiter.
			start = (   ( end > (std::string::npos - delim.size()) )
				?  std::string::npos  :  end + delim.size());
		}
	}

	//extracts the filename from the given path and stores it into <filename>
	inline std::string getFileName(std::string const & path, bool save_ext = true){

		std::string filename = path;

		// Remove directory if present.
		// Do this before extension removal in case directory has a period character.
		const size_t last_slash_idx = filename.find_last_of("\\/");
		if (std::string::npos != last_slash_idx)
			filename.erase(0, last_slash_idx + 1);

		// Remove extension if present.
		if(!save_ext)
		{
			const size_t period_idx = filename.rfind('.');
			if (std::string::npos != period_idx)
				filename.erase(period_idx);
		}

		return filename;
	}

	//string-based sprintf function
	inline std::string strprintf(const std::string fmt, ...){
		int size = 100;
		std::string str;
		va_list ap;
		while (1) {
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *)str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size) {
				str.resize(n);
				return str;
			}
			if (n > -1)
				size = n + 1;
			else
				size *= 2;
		}
		return str;
	}


	//round functions
	inline int round(float  x) { return static_cast<int>(x > 0.0f ? x + 0.5f : x - 0.5f);}
	inline int round(double x) { return static_cast<int>(x > 0.0  ? x + 0.5  : x - 0.5 );}

	//returns true if the given path is a directory
	inline bool isDirectory(std::string path){
		struct stat s;
		if( stat(path.c_str(),&s) == 0 )
		{
			if( s.st_mode & S_IFDIR )
				return true;
			else if( s.st_mode & S_IFREG )
				return false;
			else return false;
		}
		else return false;
	}

	//returns true if the given path is a file
	inline bool isFile(std::string path){
		struct stat s;
		if( stat(path.c_str(),&s) == 0 )
		{
			if( s.st_mode & S_IFDIR )
				return false;
			else if( s.st_mode & S_IFREG )
				return true;
			else return false;
		}
		else return false;
	}

	//returns true if the given string <fullString> ends with <ending>
	inline bool hasEnding (std::string const &fullString, std::string const &ending){
		if (fullString.length() >= ending.length()) {
			return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
		} else {
			return false;
		}
	}

	//returns file extension, if any (otherwise returns "")
	inline std::string getFileExtension(const std::string& FileName){
		if(FileName.find_last_of(".") != std::string::npos)
			return FileName.substr(FileName.find_last_of(".")+1);
		return "";
	}

	//number to string conversion function and vice versa
	template <typename T>
	std::string num2str ( T Number ){
		std::stringstream ss;
		ss << Number;
		return ss.str();
	}
	template <typename T>
	T str2num ( const std::string &Text ){                              
		std::stringstream ss(Text);
		T result;
		return ss >> result ? result : 0;
	}


	//make dir
	#ifdef _WIN32
	inline bool make_dir(const char* arg){
		printf("Creating directory \"%s\" ...", arg);
		bool done = _mkdir(arg) == 0;
		bool result = done || errno != ENOENT;
		printf("%s\n", result? "DONE!" : "ERROR!");
		return result;
	}
	#else
	inline bool make_dir(const char* arg){
		printf("Creating directory \"%s\" ...", arg);
		bool done = mkdir(arg,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0;
		bool result = done || errno == EEXIST;
		printf("%s\n", result? "DONE!" : "ERROR!");
		return result;
	}
	#endif


	//file deleting
	#ifdef _WIN32
	inline void delete_file( const char* arg ){
		if(system(strprintf("del /F /Q /S \"%s\"", arg).c_str())!=0)
			fprintf(stderr,"Can't delete file \"%s\"\n", arg);
	}
	#else
	inline void delete_file( const char* arg ){
		if(system(strprintf("rm -f \"%s\"", arg).c_str())!=0)
			fprintf(stderr,"Can't delete file \"%s\"\n", arg);
	}
	#endif

	//cross-platform current function macro
	#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600))
	# define _aia_current_function __PRETTY_FUNCTION__
	#elif defined(__DMC__) && (__DMC__ >= 0x810)
	# define _aia_current_function __PRETTY_FUNCTION__
	#elif defined(__FUNCSIG__)
	# define _aia_current_function __FUNCSIG__
	#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
	# define _aia_current_function __FUNCTION__
	#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
	# define _aia_current_function __FUNC__
	#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
	# define _aia_current_function __func__
	#else
	# define _aia_current_function "(unknown)"
	#endif
	/*-------------------------------------------------------------------------------------------------------------------------*/


	/*******************************
	*    EXCEPTIONS DEFINITIONS    *
	********************************
	---------------------------------------------------------------------------------------------------------------------------*/
	class error
	{
		private:

			std::string source;
			std::string message;
			failure_type type;
			error(void);

		public:

			error(std::string _message, std::string _source = "unknown", failure_type _type = UNDEFINED){
				source = _source; message = _message; type = _type;}
			~error(void){}
			const char* what() const {return message.c_str();}
			const char* getSource() const {return source.c_str();}
			failure_type getType() {return type;}
	};


	/*******************************************
	*   OpenCV UTILITY functions	           *
	********************************************
	---------------------------------------------------------------------------------------------------------------------------*/
	
	// wrapper that adapts cv::imshow to the current screen
	inline void imshow(const std::string winname, cv::InputArray arr, bool wait = true, float scale = 1.0)
	{
		// create window
		cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);

		// resize window so as to fit screen size while maintaining image aspect ratio
		int win_height = arr.size().height, win_width = arr.size().width;

		cv::resizeWindow(winname, round(win_width*scale), round(win_height*scale));

		// display image
		cv::imshow(winname, arr);

		// wait for key pressed
		if(wait)
			cv::waitKey(0);
	}


	// convert OpenCV depth (which is a macro) to the corresponding bitdepth
	inline int bitdepth(int ocv_depth)
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


	// generic frame-by-frame video processing function
	inline void processVideoStream(
		const std::string & inputPath = "",						// input video path (if empty, camera will be used)
		cv::Mat (*frameProcessor)(const cv::Mat& frame) = 0,	// frame-by-frame image processing function
		const std::string & outputPath = "",					// output video path (optional)
		bool showOnlyProcessedStream = false,
		int delay = 0,
		float rescale = 0)
	
	{
		// open input video stream (either from camera or from file)
		cv::VideoCapture capture;
		inputPath.empty() ? capture.open(0) : capture.open(inputPath);
		if (!capture.isOpened())
			throw ipa::error(ipa::strprintf("Cannot open input video stream from %s", inputPath.empty() ? "camera" : inputPath.c_str()));


		// get frame rate (also known as Frames Per Second)
		double fps = inputPath.empty() ? CAMERA_FPS : capture.get(cv::CAP_PROP_FPS);


		// open output stream (if required)
		cv::VideoWriter output;
		if(!outputPath.empty())
		{
			output.open(outputPath, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size((int)(capture.get(cv::CAP_PROP_FRAME_WIDTH)),(int)(capture.get(cv::CAP_PROP_FRAME_HEIGHT))), true);
			if (!output.isOpened())
				throw ipa::error("Cannot open output video stream");
		}

		// for all frames in video
		bool stop(false);
		if(!delay)
			delay = inputPath.empty()? 1 : static_cast<int>(1000/fps); // delay between each frame in ms
		while (!stop) 
		{
			// read next frame if any
			cv::Mat frame;
			if (!capture.read(frame))
				break;

			// process frame
			cv::Mat processedFrame = frameProcessor ? frameProcessor(frame) : frame.clone();

			// display original and processed frame
			if (!showOnlyProcessedStream)
			{
				if (rescale)
					cv::resize(frame, frame, cv::Size(0, 0), rescale, rescale);
				cv::imshow("Original stream", frame);
			}
			if(rescale)
				cv::resize(processedFrame, processedFrame, cv::Size(0,0), rescale, rescale);
			cv::imshow("Processed stream",processedFrame);

			// write output frame
			if(output.isOpened())
				output.write(processedFrame);

			// introduce a delay
			if (cv::waitKey(delay)>=0)
			{
				stop= true;
				cv::destroyWindow("Processed stream");
			}
		}
	}


	// return a vector of rectangles containing the detected faces on the given image frame
	inline std::vector < cv::Rect > faceDetector(const cv::Mat& frame) 
	{
		if(!ipa::isFile(FACE_DETECTOR_PATH))
			throw ipa::error(ipa::strprintf("Cannot load face detector from \"%s\"", FACE_DETECTOR_PATH.c_str()));

		//create the cascade classifier object used for the face detection
		static cv::CascadeClassifier face_cascade(FACE_DETECTOR_PATH);

		// create frame copy
		cv::Mat frameCopy = frame.clone();

		// convert captured image to gray scale (if needed) and equalize
		if (frame.channels() == 3)
			cv::cvtColor(frameCopy, frameCopy, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(frameCopy, frameCopy);

		// create a vector array to store the faces found
		std::vector<cv::Rect> faces;

		// find faces and store them in the vector array
		face_cascade.detectMultiScale(frameCopy, faces, 1.1, 3, cv::CASCADE_FIND_BIGGEST_OBJECT|cv::CASCADE_SCALE_IMAGE, cv::Size(30,30));

		return faces;
	}
	/*-------------------------------------------------------------------------------------------------------------------------*/




	/***********************************************
	*    DEBUG, WARNING and EXCEPTION FUNCTIONS    *
	************************************************
	---------------------------------------------------------------------------------------------------------------------------*/
	inline void warning(const char* message, const char* source = 0){
		if(source)
			printf("\n**** WARNING (source: \"%s\") ****\n"
			"    |=> \"%s\"\n\n", source, message);
		else
			printf("\n**** WARNING ****: %s\n", message);
	}

	inline void debug(debug_level dbg_level, const char* message=0, const char* source=0){
		if(DEBUG >= dbg_level){
			if(message && source)
				printf("\n---- DEBUG (level %d, source: \"%s\") ----\n"
				         "    |=> \"%s\"\n\n", dbg_level, source, message);
			else if(message)
				printf("\n---- DEBUG (level %d) ----: %s\n", dbg_level, message);
			else if(source)
				printf("\n---- DEBUG (level %d) ----: in \"%s\"\n", dbg_level, source);
		}
	}
	/*-------------------------------------------------------------------------------------------------------------------------*/
}

#endif /* _config_h */