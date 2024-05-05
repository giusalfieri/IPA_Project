#ifndef _UNICAS_CONFIG_H
#define _UNICAS_CONFIG_H

#define NOMINMAX

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "ucasFileUtils.h"
#include "ucasStringUtils.h"
#include "ucasMultithreading.h"
#include "ucasMathUtils.h"
#include "ucasImageUtils.h"
#include "ucasBreastUtils.h"
#include "ucasMachineLearningUtils.h"
#include "ucasExceptions.h"
#include "ucasLog.h"
#include "ucasStringUtils.h"
#include "ucasTypes.h"

namespace ucas
{
	static const char* VERSION = "1.2.1";		//software version
	const int STATIC_BIG_STRING_SIZE = 3000;	//size of big static C-strings
	const int STATIC_SMALL_STRING_SIZE = 50;	//size of small static C-string
	const int TEXT_FILE_LINE_SIZE = 50000;		//size of buffers used to store each line when parsing text files
	const int UNDEFINED = -1;					//undefined constant

	template <typename T>
	T* mallocinit(const int N, T val){
		T* o = new T[N];
		for(int i=0; i<N; i++)
			o[i] = val;
		return o;
	}
}

/**************************************************************************************************************************
 *   Cross-platform UTILITY functions	   																			      *
 **************************************************************************************************************************/

//3-maximum
#define MAX3(a,b,c) (a < b ? MAX(b,c) : MAX(a,c))

//sign function
#define SGN( arg )	( (arg) < 0 ? -1 : 1 )

//sign function for boolean (true = +1, false = -1)
#define BOOL_TO_SGN( arg )( (arg)? 1 : -1 )

//assigns 0 to the given argument if it is less than zero
#define EV_ZERO_THRESHOLD(arg) ((arg)<0 ? 0 : (arg))

//ROUND
#define ROUND_FLOAT2INT(x) ( static_cast<int>((x) > 0.0f ? (x) + 0.5f : (x) - 0.5f) )
#define ROUND_DOUBLE2INT(x) ( static_cast<int>(x > 0.0 ? (x) + 0.5f : (x) - 0.5) )

//file deleting
#ifdef _WIN32
#define RM_FILE( arg ) 						\
	sprintf(ucas::sys_cmd, "del /F /Q /S \"%s\"", arg);  \
	if(system(ucas::sys_cmd)!=0)					\
	fprintf(stderr,"Can't delete file \"%s\"\n", arg);
#else
#define RM_FILE( arg )						\
	sprintf(ucas::sys_cmd, "rm -f \"%s\"", arg);		\
	if(system(ucas::sys_cmd)!=0)					\
	fprintf(stderr,"Can't delete file \"%s\"\n", arg);
#endif

//recycling
#ifdef _WIN32
#define RECYCLE( arg ) 						\
	sprintf(ucas::sys_cmd, "recycle \"%s\"", (arg));  \
	if(system(ucas::sys_cmd)!=0)					\
	fprintf(stderr,"Can't recycle file \"%s\"\n", (arg));
#else
#define RECYCLE( arg ) \
	fprintf(stderr,"RECYCLE macro not defined on this system\n");
#endif

//"PAUSE" function
#ifdef _WIN32
#define system_PAUSE() system("PAUSE"); std::cout<<std::endl;
#define system_CLEAR() system("cls");
#else
#define system_CLEAR() system("clear");
#define system_PAUSE()										\
	std::cout<<"\n\nPress RETURN key to continue..."<<std::endl<<std::endl;	\
	std::cin.get(); 
#endif

//FILE COPY
#ifdef _WIN32
#define COPY_ASCII_FILE( src, dst ) 						\
	sprintf(ucas::sys_cmd, "copy /Y /A \"%s\" \"%s\"", src, dst);  \
	if(system(ucas::sys_cmd)!=0)					\
	fprintf(stderr,"Can't copy file from \"%s\" to \"%s\"\n", src, dst);
#else
#define COPY_ASCII_FILE( src, dst ) \
	fprintf(stderr,"COPY_ASCII_FILE macro not defined on this system\n");
#endif
#ifdef _WIN32
#define COPY_BINARY_FILE( src, dst ) 						\
	sprintf(ucas::sys_cmd, "copy /Y /B \"%s\" \"%s\"", src, dst);  \
	if(system(ucas::sys_cmd)!=0)					\
	fprintf(stderr,"Can't copy file from \"%s\" to \"%s\"\n", src, dst);
#else
#define COPY_BINARY_FILE( src, dst ) \
	fprintf(stderr,"COPY_BINARY_FILE macro not defined on this system\n");
#endif

//FOLDER COPY
#ifdef _WIN32
#define COPY_FOLDER( src, dst ) 						\
	sprintf(ucas::sys_cmd, "xcopy /Y /S /Q /I \"%s\" \"%s\"", src, dst);  \
	if(system(ucas::sys_cmd)!=0)					\
	fprintf(stderr,"Can't copy directory from \"%s\" to \"%s\"\n", src, dst);
#else
#define COPY_FOLDER( src, dst ) \
	fprintf(stderr,"COPY_FOLDER macro not defined on this system\n");
#endif

#endif //_MCD_CONFIG_H
