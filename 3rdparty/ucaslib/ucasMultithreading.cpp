#include "ucasMultithreading.h"

namespace ucas
{
	// set global options with default values
	int THREADS_CONCURRENCY = std::thread::hardware_concurrency();			//number of concurrent threads when multithread mode is enabled
	bool MULTITHREADED_TESTING = true;			//enable multithreaded testing routines
	bool MULTITHREADED_TRAINING = true;			//enable multithreaded training routines
	bool MULTITHREADED_FEATURES = true;			//enable multithreaded feature extraction routines
}