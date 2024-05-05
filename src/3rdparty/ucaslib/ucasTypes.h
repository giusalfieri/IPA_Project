#ifndef _UNICAS_TYPES_H
#define _UNICAS_TYPES_H

#include <vector>
#include <string>

namespace ucas
{
    /*******************
    *       TYPES      *
    ********************
    ---------------------------------------------------------------------------------------------------------------------------*/
	typedef signed char	sint8;					//8-bit  signed   integers (-128                       -> +127)
	typedef short sint16;						//16-bit signed   integers (-32,768                    -> +32,767)
	typedef int sint32;							//32-bit signed   integers (-2,147,483,648             -> +2,147,483,647)
	typedef long long sint64;					//64-bit signed   integers (–9,223,372,036,854,775,808 -> +9,223,372,036,854,775,807)
	typedef unsigned char uint8;				//8-bit  unsigned integers (0 -> +255)
	typedef unsigned short int uint16;			//16-bit unsigned integers (0 -> +65,535)
	typedef unsigned int uint32;				//32-bit unsigned integers (0 -> +4,294,967,295)
	typedef unsigned long long uint64;			//64-bit unsigned integers (0 -> +18,446,744,073,709,551,615
	typedef float real32;						//real single precision
	typedef double real64;						//real double precision
	typedef real32 data_t;						//CRITICAL: sample data type affects correctness, operations precision and RAM usage	
	const int pixel_t = CV_32FC1;				//CRITICAL: OpenCV type for sample data
	typedef real32 weight_t;					//data type for sample weights
	typedef enum {
		NO      = 0,
		YES     = 1,
		DONTCARE = -1
	} TriState;
	inline TriState str2TriState(const std::string & str){
		if(str.compare("NO") == 0)
			return NO;
		else if(str.compare("YES") == 0)
			return YES;
		else return DONTCARE;
	}

	typedef std::vector < std::pair < std::string, std::string > > paired_strings;
	typedef std::pair < double, double > pair_real;
}
#endif