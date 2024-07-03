#ifndef _UNICAS_MULTITHREADING_H
#define _UNICAS_MULTITHREADING_H

#include <thread>
#include <mutex>
#include <condition_variable>

namespace ucas
{
	// global options
	extern int	 THREADS_CONCURRENCY;			// number of concurrent threads when multithreading is enabled
	extern bool  MULTITHREADED_TESTING;			// enable multithreaded testing routines
	extern bool  MULTITHREADED_TRAINING;		// enable multithreaded training routines
	extern bool  MULTITHREADED_FEATURES;		// enable multithreaded feature extraction routines
	
	
	// for thread synchronization
	class Barrier
	{
		private:

			std::mutex _mutex;
			std::condition_variable _cv;
			std::size_t _count;

		public:

			explicit Barrier(std::size_t count) : _count(count) { }
			void wait()
			{
				std::unique_lock<std::mutex> lock(_mutex);
				if (--_count == 0) {
					_cv.notify_all();
				} else {
					_cv.wait(lock, [this] { return _count == 0; });
				}
			}
	};
}

#endif