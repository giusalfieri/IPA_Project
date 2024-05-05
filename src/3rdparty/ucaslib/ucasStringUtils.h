#ifndef _UNICAS_STRING_UTILS_H_
#define _UNICAS_STRING_UTILS_H_

#include <string>
#include <algorithm>
#include <cstdarg>
#include <sstream>
#include <vector>
#include <cstring>
#include <regex>
#include "ucasExceptions.h"
//#include "ucasMathUtils.h"

namespace ucas
{
	//returns a new string from "string" by replacing all occurrences of "c1" with "c2"
	inline std::string strrpl(const char* str, char c1, char c2)
	{
		std::string s = str;
		std::replace( s.begin(), s.end(), c1, c2); // replace all 'x' to 'y'
		return s;
	}

	//replaces in place all occurrences of "oldStr" with "newStr" in "str"
	inline std::string strrpl(std::string& str, const std::string& oldStr, const std::string& newStr)
	{
		size_t pos = 0;
		while((pos = str.find(oldStr, pos)) != std::string::npos)
		{
			str.replace(pos, oldStr.length(), newStr);
			pos += newStr.length();
		}
		return str;
	}

	//replaces all occurrences of "c1" with "c2" in-place
	inline void strirpl(std::string & str, char c1, char c2)
	{
		std::replace( str.begin(), str.end(), c1, c2); // replace all 'x' to 'y'
	}

	//replaces all occurrences of "c1" with "c2" in-place
	inline void strirpl(char* str, char c1, char c2)
	{
		for(size_t i=0; i<strlen(str); i++)
			if(str[i] == c1)
				str[i] = c2;
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

	//the case insensitive version of C strstr() function
	inline static const char* stristr(const char *str1, const char *str2)
	{
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
	inline int stricmp (const char *s1, const char *s2)	
	{
		if (s1 == NULL) return s2 == NULL ? 0 : -(*s2);
		if (s2 == NULL) return *s1;

		char c1, c2;
		while ((c1 = tolower (*s1)) == (c2 = tolower (*s2)))
		{
			if (*s1 == '\0') break;
			++s1; ++s2;
		}

		return c1 - c2;
	}

	inline static std::string int2str(const int& val)
	{
		std::stringstream ss;
		ss << val;
		return ss.str();
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

	inline static std::string list2str( std::vector< std::string> alist )
	{
		std::string res = "{";
		for(int i=0; i<alist.size(); i++)
			res += "\"" + alist[i] + "\"" + (i == alist.size()-1 ? "}" : ", ");
		return res;
	}

	//fgetstr() - mimics behavior of fgets(), but removes new-line character at end of line if it exists
	inline static char *fgetstr(char *string, int n, FILE *stream)	
	{
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

	// removes duplicate spaces from the given string
	// *** optional *** : also removes initial character if it is a space, i.e. the result string does not begin with a space
	inline bool bothAreSpaces(char lhs, char rhs) { return (lhs == rhs) && (lhs == ' '); }
	inline std::string singlespaces(const std::string & _str, bool no_begin_space = true, bool no_end_space = true)
	{
		std::string str = _str;
		std::string::iterator new_end = std::unique(str.begin(), str.end(), bothAreSpaces);
		str.erase(new_end, str.end()); 

		if(no_begin_space && str[0] == ' ')
			str = str.substr(1, str.size()-1);
		if(no_end_space && str[str.size()-1] == ' ')
			str = str.substr(0, str.size()-1);

		return str;
	}

	// removes carriage return characters
	inline std::string clcr(const std::string & _str){
		std::string str = _str;
		str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
		str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
		return str;
	}	


	inline void	split(const std::string& theString, std::string delim, std::vector<std::string>& tokens)
	{
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

	//returns true if the given string <fullString> ends with <ending>
	inline bool hasEnding (std::string const &fullString, std::string const &ending){
		if (fullString.length() >= ending.length()) {
			return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
		} else {
			return false;
		}
	}

	// removes all tab, space and newline characters from the given string (in-place version and copy-based version)
	inline std::string clsi(std::string& string){
		string.erase(std::remove(string.begin(), string.end(), '\t'), string.end());
		string.erase(std::remove(string.begin(), string.end(), ' '),  string.end());
		string.erase(std::remove(string.begin(), string.end(), '\n'), string.end());
		string.erase(std::remove(string.begin(), string.end(), '\r'), string.end());
		return string;
	}		
	inline std::string cls(std::string string){
		string.erase(std::remove(string.begin(), string.end(), '\t'), string.end());
		string.erase(std::remove(string.begin(), string.end(), ' '),  string.end());
		string.erase(std::remove(string.begin(), string.end(), '\n'), string.end());
		string.erase(std::remove(string.begin(), string.end(), '\r'), string.end());
		return string;
	}	

	// string shortening
	inline std::string shorten(const std::string & str, size_t max_length)
	{
		if(str.size() <= max_length)
			return str;
		else
		{
			std::string res = str.substr(0, max_length-3);
			res = res + "...";
			return res;
		}
	}

	// string padding (to the right)
	inline std::string padding(const std::string & str, size_t fixed_dim, char c = ' ')
	{
		std::string res = str;
		while(res.size() < fixed_dim)
			res += c;
		return res;
	}

	

	// parse a string of 'delim'-separated (numeric) values of type 'T' and put them into a vector
	template <typename T>
	std::vector<T> str2numlist(const std::string & str, std::string delim = ",")
	{
		std::vector<std::string> tokens;
		split(str, delim, tokens);
		std::vector<T> numlist;
		for(int i=0; i<tokens.size(); i++)
			numlist.push_back(str2num<T>(tokens[i]));
		return numlist;
	}

	template <typename T>
	std::string numlist2str(const std::vector<T> numlist, std::string delim = ","){
		std::string str;
		for(size_t k=0; k<numlist.size(); k++)
			str += num2str<T>(numlist[k]) + (k < numlist.size() - 1 ? "," : "");
		return str;
	}

	// parse a range in the format [a,b)\[c,d)
	inline static void parse_range(const std::string & str, int &a, int &b, int &c, int &d) 
	{
		std::regex rgx("^\\[[0-9]+,([0-9]+|inf)\\)\\\\[[0-9]+,([0-9]+|inf)\\)$");
		if(!std::regex_match(str, rgx))
			throw Error(strprintf("\"%s\" does not match string pattern [a,b)\\[c,d)", str.c_str()));

		std::string left  = str.substr(0, str.find("\\"));
		std::string astr = left.substr(0, left.find(","));
		astr = astr.substr(1);
		std::string bstr = left.substr(left.find(",")+1);
		bstr = bstr.substr(0, bstr.length()-1);

		std::string right = str.substr(str.find("\\")+1);
		std::string cstr = right.substr(0, right.find(","));
		cstr = cstr.substr(1);
		std::string dstr = right.substr(right.find(",")+1);
		dstr = dstr.substr(0, dstr.length()-1);

		a = astr == "inf" ? std::numeric_limits<int>::max() : ucas::str2num<int>(astr);
		b = bstr == "inf" ? std::numeric_limits<int>::max() : ucas::str2num<int>(bstr);
		c = cstr == "inf" ? std::numeric_limits<int>::max() : ucas::str2num<int>(cstr);
		d = dstr == "inf" ? std::numeric_limits<int>::max() : ucas::str2num<int>(dstr);
	}
}

#endif