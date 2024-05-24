#include "template_extraction.h"
#include "k_mean.h"

//#define USE_EXTRACT_TEMPLATES


int main()
{
	
	//#ifdef USE_EXTRACT_TEMPLATES
	extractTemplates();
	
	//#endif

	kMeansClustering_BySize();
	
	kMeansClustering_ByIntensity();

	return EXIT_SUCCESS;
}
