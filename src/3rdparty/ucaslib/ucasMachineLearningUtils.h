#pragma once

#include <string>
#include <fstream>
#include <functional>
#include <algorithm>
#include <map>
#include "ucasConfig.h"

namespace ucas
{
	/*************************
	*   CLASSES              *
	**************************
	---------------------------------------------------------------------------------------------------------------------------*/
	template <typename T>
	struct ROCpoint
	{
		T tpr, fpr;		// point
		T t;			// threshold
		bool assigned;	// this point has been assigned to a certain worker thread (for multithread processing)

		T tpr_sum_sq;	// accumulation of sum of squared tprs (for averaged ROCs)
		T tpr_std;		// standard deviation of tpr (for averaged ROCs)

		size_t P, N;	// number of positive (P) and negative (N) test samples from which this point has been calculated

		ROCpoint<T>(T _tpr, T _fpr, T _t, size_t _P = 0, size_t _N=0) : tpr(_tpr), fpr(_fpr), t(_t), assigned(false), tpr_sum_sq(0), tpr_std(0), P(_P), N(_N){}
		ROCpoint<T>() : tpr(0), fpr(0), t(0), assigned(false), tpr_sum_sq(0), tpr_std(0), P(0), N(0){}

		bool operator==(const ROCpoint<T> & r){
			return tpr == r.tpr && fpr == r.fpr;
		}
		bool operator!=(const ROCpoint<T> & r){
			return !(this == r);
		}

		T accuracy() const{
			T TP = tpr * P;
			T FP = fpr * N;
			T TN = N - FP;
			return (TP+TN)/(P+N);
		}

		T F1() const{
			T TP = tpr * P;
			T FP = fpr * N;
			T FN = P - TP;
			return (2*TP)/(2*TP + FP + FN);
		}

		T MCC() const{
			T TP = tpr * P;
			T FP = fpr * N;
			T TN = N - FP;
			T FN = P - TP;
			return (TP*TN-FP*FN)/std::sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
		}
	};

	// utility function: computes TP, FP, TN, FN from binary class samples
	template<typename T>
	void eval(
		const std::vector <T> & pos, 
		const std::vector <T> & neg,
		T & TP, T & FP, T & TN, T & FN,
		bool pos_greater_than_neg = true, 
		T threshold = 0) 
	
	{
		TP = 0, FP = 0;
		for(auto & p : pos)
			if(pos_greater_than_neg ? p >= threshold : p <= threshold)
				TP++;
		for(auto & n : neg)
			if(pos_greater_than_neg ? n >= threshold : n <= threshold)
				FP++;
		TN = neg.size() - FP;
		FN = pos.size() - TP;
	}


	/*************************
	*   TYPES and CONSTANTS  *
	**************************
	---------------------------------------------------------------------------------------------------------------------------*/
	typedef std::vector< std::string > stringlist;
	typedef std::vector< ROCpoint<double> > ROC_d;
	typedef std::vector< ROC_d > ROC_2d;
	typedef std::vector< ROC_2d > ROC_3d;
	typedef std::vector<double> vector_d;
	typedef std::vector< std::vector<double> > vector_2d;
	typedef std::map< int, std::map<int, double> > map_2d;

	
	/*************************
	*   FUNCTIONS            *
	**************************
	---------------------------------------------------------------------------------------------------------------------------*/

	// generate distinct ROC points from sample scores
	template <typename T>
	inline std::vector <T>
		ROC_points(
		const std::vector<T> & pos,		// positive sample score array
		const std::vector<T> & neg,		// negative sample score array
		bool pos_greater_than_neg = 1,	// 1(0) = the higher(lower) the sample score, the higher the probability of being positive
		bool exact = true,				// true: generate all points; false: generate a subset of ROC points by using the minority class samples
		int precision = -1,				// maximum number of decimal digits of thresholds (default unlimited: -1)
		bool verbose = false,			// verbose mode
		const std::string & desc = ""	// (verbose mode only) sample description
		) 
	{
		// check preconditions
		if(pos.empty())
			UCAS_THROW(ucas::strprintf("in generate_roc_points(%s): no positive samples found", desc.c_str()));
		if(neg.empty())
			UCAS_THROW(ucas::strprintf("in generate_roc_points(%s): no negative samples found", desc.c_str()));

		// generate unique, left-to-right ROC points from distinct scores
		std::vector<T> points;
		if(exact)
		{
			for(int i=0; i<pos.size(); i++)
				if(!ucas::is_nan(pos[i]))
					points.push_back(pos[i]);
			for(int i=0; i<neg.size(); i++)
				if(!ucas::is_nan(neg[i]))
					points.push_back(neg[i]);
		}
		else
		{
			const std::vector<T> & majority_class_scores = pos.size() < neg.size() ? pos : neg;
			for(int i=0; i<majority_class_scores.size(); i++)
				if(!ucas::is_nan(majority_class_scores[i]))
					points.push_back(majority_class_scores[i]);
		}
		if(verbose)
			printf("   in generate_roc_points(%s): number of points = %d %s\n", desc.c_str(), int(points.size()), exact == false ? "(sampled from minority class)" : "");

		// order
		if(pos_greater_than_neg)
			std::sort(points.begin(), points.end(), std::greater<T>());	
		else
			std::sort(points.begin(), points.end(), std::less<T>());

		// decrease precision when needed
		if(precision >= 0)
		{
			points.erase(unique(points.begin(), points.end()), points.end());
			if(verbose)
			{
				printf("   in generate_roc_points(%s): after removing duplicates,  number of points = %d\n", desc.c_str(), int(points.size()));
				printf("   in generate_roc_points(%s): decrease precision (number of significant digits = %d)...", desc.c_str(), precision);
			}

			for(size_t i=0; i<points.size(); i++)
				if(points[i] != std::numeric_limits<T>::infinity() && points[i] != -std::numeric_limits<T>::infinity())
					points[i] = ucas::str2num<double>(strprintf((std::string("%.") + ucas::num2str<int>(precision) + "f").c_str(), points[i]));
			if(verbose)
				printf("   done\n");
		}

		// remove duplicates
		points.erase(unique(points.begin(), points.end()), points.end());
		if(verbose)
			printf(  "   in generate_roc_points(%s): after removing duplicates,  number of points = %d\n", desc.c_str(), int(points.size()));

		return points;
	}


	// calculate ROC curve from positive and negative sample score arrays
	// fastest algorithm for *single-thread* computation of *exact* ROC curve
	template <typename T>
	struct sample
	{
		T score;
		bool positive;
		friend bool operator<(const sample& l, const sample& r){ return l.score < r.score;}
		friend bool operator>(const sample& l, const sample& r){ return r < l;}
	};
	template <typename T> 
	inline 
		std::vector< ROCpoint<T> >					// return a sequence of ROC points
		ROC_compute(
		const std::vector<T> & pos,					// positive sample score array
		const std::vector<T> & neg,					// negative sample score array
		bool pos_greater_than_neg = 1,				// 1(0) = the higher(lower) the sample score, the higher the probability of being positive
		std::vector<T> points = std::vector<T>(),	// ROC points (thresholds) to compute given by descending(ascending) order if pos_greater_than_neg is true(false)
		bool verbose = false)
		
	{
		// check preconditions
		if(pos.empty())
			UCAS_THROW("in ROC_compute(): no positive samples found");
		if(neg.empty())
			UCAS_THROW("in ROC_compute(): no negative samples found");
		for(size_t i=0; i<pos.size(); i++)
			if(ucas::is_nan(pos[i]))
				UCAS_THROW(ucas::strprintf("in ROC_compute(): found nan at positive sample %d", i));
		for(size_t i=0; i<neg.size(); i++)
			if(ucas::is_nan(neg[i]))
				UCAS_THROW(ucas::strprintf("in ROC_compute(): found nan at negative sample %d", i));
		for(size_t i=0; i<points.size(); i++)
			if(ucas::is_nan(points[i]))
				UCAS_THROW(ucas::strprintf("in ROC_compute(): found nan at ROC point %d", i));
		for(size_t i=0; pos_greater_than_neg && points.size() && i<points.size()-1; i++)
			if( points[i] < points[i+1]) 
				UCAS_THROW("in ROC_compute(): ROC points not in descending order");
		for(size_t i=0; !pos_greater_than_neg && points.size() && i<points.size()-1; i++)
			if( points[i] > points[i+1]) 
				UCAS_THROW("in ROC_compute(): ROC points not in ascending order");


		// create sample vector
		std::vector <sample<T> > samples(pos.size() + neg.size());
		for(size_t i=0; i<pos.size(); i++)
		{
			if(!ucas::is_nan(pos[i]))	
			{
				samples[i].score = pos[i];
				samples[i].positive = true;
			}
		}
		for(size_t i=0; i<neg.size(); i++)
		{
			if(!ucas::is_nan(neg[i]))
			{
				samples[i+pos.size()].score = neg[i];
				samples[i+pos.size()].positive = false;
			}
		}

		// order sample by decreasing (or increasing) scores
		if(pos_greater_than_neg)
			std::sort(samples.begin(), samples.end(), std::greater< sample<T> >());	
		else
			std::sort(samples.begin(), samples.end(), std::less< sample<T> >());

		// calculate ROC
		std::vector< ROCpoint<T> > out;
		double TP=0, FP=0, P=pos.size(), N=neg.size();

		// modality 1 - all ROC points (exact ROC)
		if(points.empty())
		{
			T fprev = pos_greater_than_neg ? inf<T>() : -inf<T>();
			for(size_t i=0; i<samples.size(); i++)
			{
				if(samples[i].score != fprev)
				{	
					out.push_back(ROCpoint<T>(TP/P, FP/N, fprev, P, N));
					fprev = samples[i].score;
				}
				if(samples[i].positive)
					TP++;
				else
					FP++;
			}
			out.push_back(ROCpoint<T>(TP/P, FP/N, fprev, P, N));
		}
		// modality 2 - only ROC points given as input (approximated ROC)
		else
		{
			// precondition check
			if ( pos_greater_than_neg )
			{
				if( points.front() > samples.front().score ||  points.back() < samples.back().score )
					UCAS_THROW(ucas::strprintf("in ROC_compute(): decreasing ROC range [%f,%f] is not within sample score range [%f,%f]", points.front(), points.back(), samples.front().score, samples.back().score));
			}
			else
			{
				if( points.front() < samples.front().score ||  points.back() > samples.back().score ) 
					UCAS_THROW(ucas::strprintf("in ROC_compute(): increasing ROC range [%f,%f] is not within sample score range [%f,%f]", points.front(), points.back(), samples.front().score, samples.back().score));
			}

			out.resize(points.size());
			size_t j=0;
			for(size_t i=0; i<samples.size(); i++)
			{
				if(pos_greater_than_neg ? samples[i].score < points[j] : samples[i].score > points[j])
				{
					if(j < points.size() - 1)
					{
						out[j] = ROCpoint<T>(TP/P, FP/N, points[j], P, N);
						j++;
					}
				}
				if(samples[i].positive)
					TP++;
				else
					FP++;
			}
			if(j < points.size())
				out[j] = ROCpoint<T>(TP/P, FP/N, points[j], P, N);
		}

		// push (0,0) at front if needed
		if(out.front().tpr != 0 || out.front().fpr != 0)
			out.insert(out.begin(), ROCpoint<T>(0,0,pos_greater_than_neg ? inf<T>() : -inf<T>(), P, N));

		// push (1,1) at back if needed
		if(out.back().tpr != 1 || out.back().fpr != 1)
			out.push_back(ROCpoint<T>(1,1,pos_greater_than_neg ? -inf<T>() : inf<T>(), P, N));


		return out;
	}

	// get tpr by interpolation
	template <typename T>
	inline T
		ROC_interp_tpr(
		const std::vector < ROCpoint<T> > & roc,
		T fpr
		) 
	{
		// get closest point from the right
		ROCpoint<T> right = roc.back();
		for(int j=int(roc.size())-2; j >= 0; j--)
			if(roc[j].fpr > fpr)
				right = roc[j];
			else 
				break;

		// get closest point from the left
		ROCpoint<T> left  = roc.front();
		for(int j=1; j<roc.size(); j++)
			if(roc[j].fpr <= fpr)
				left = roc[j];
			else 
				break;

		// same fpr = return
		if(right.fpr == left.fpr)
			return std::max(left.tpr, right.tpr);
		// otherwise linear interpolation
		else
			return left.tpr + (right.tpr - left.tpr)* ((fpr - left.fpr)/(right.fpr - left.fpr));
	}

	// worker for ROC averaging along tpr axis
	inline void
		ROC_average_worker(
		Barrier & barrier,						// (INPUT)  thread synchronization barrier
		ROC_2d & rocs,							// (INPUT)  rocs to average
		ROC_d & avg_roc,						// (OUTPUT) averaged roc (preallocated with precomputed fpr points)
		bool & error,							// (OUTPUT) set to true when error occurs
		bool verbose = false)					// (option) verbose mode
	{
		// instance mutex
		static std::mutex id_mutex, iter_mutex, error_mutex;

		// instance shared variables
		static int jobs = int(avg_roc.size());
		static int ids = 0;

		// reset shared variables
		jobs = int(avg_roc.size());
		ids = 0;
		barrier.wait();

		// get thread id as integer starting from 0 (first thread spawn)
		id_mutex.lock();
		int worker = ids++;
		id_mutex.unlock();

		// hello world
		if(verbose)
			printf("   in ROC_average_worker(%03d): hello world\n", worker);
		ucas::Timer timer_local, timer_global;


		// try-catch block to detect exceptions and stop the entire process if needed
		int job = 0;
		int nPoints = int(avg_roc.size());
		try
		{
			// check preconditions
			if(rocs.empty())
				UCAS_THROW("empty input rocs");
			if(avg_roc.empty())
				UCAS_THROW("average roc has not been allocated");

			// go on until a new job is available and no errors occurred
			while(job >= 0 && !error)
			{
				// get next job available
				iter_mutex.lock();
				job = --jobs;
				iter_mutex.unlock();

				if(job >= 0)
				{
					if(verbose && job%10000 == 0 && job != 0)
						printf("   in ROC_average_worker(%03d) [job %d of %d]: time elapsed = %.0f s, ETA = %.0f s\n", worker, nPoints-job, nPoints, timer_global.elapsed<float>(), (timer_global.elapsed<float>() * job)/(nPoints-job));

					int k = nPoints-job-1;
					double tpr = 0;
					double tpr_sq = 0;
					for(size_t i=0; i<rocs.size(); i++)
					{
						double tpr_int = ROC_interp_tpr<double>(rocs[i], avg_roc[k].fpr);
						tpr += tpr_int;
						tpr_sq += tpr_int*tpr_int;

						avg_roc[k].P += rocs[i][0].P;
						avg_roc[k].N += rocs[i][0].N;
					}

					// store result (mutual exclusion guaranteed by 'job')
					avg_roc[k].tpr = tpr/rocs.size();
					double sqrt_arg = tpr_sq/rocs.size() - std::pow(tpr/rocs.size(), 2.0f);
					if(sqrt_arg > 0)
						avg_roc[k].tpr_std = std::sqrt(sqrt_arg);
					else
						avg_roc[k].tpr_std = 0;
					avg_roc[k].P = ucas::round(avg_roc[k].P / double(rocs.size()));
					avg_roc[k].N = ucas::round(avg_roc[k].N / double(rocs.size()));
				}
			}
		}
		catch (ucas::Error & e)
		{
			error_mutex.lock();
			error = true;
			printf("   in ROC_average_worker(%03d) [job %d of %d]: ERROR : %s\n", worker, nPoints-job, nPoints, e.what());
			error_mutex.unlock();
		}
	}


	//  ROC averaging
	inline void
		ROC_average(
		ROC_2d & rocs,							// (INPUT)  rocs to average
		ROC_d & avg_roc,						// (OUTPUT) averaged roc (preallocated)
		int j = 1,							    // (option) number of concurrent threads
		bool small = true,						// (option) ROC averaging is done at fixed FPR points (small = true) instead that at all FPR points available in 'rocs' (small = false)
		bool verbose = false)					// (option) verbose mode
		
	{
		// check preconditions
		if(rocs.empty())
			UCAS_THROW("empty input rocs");


		// compute fixed FPR points
		std::vector <double> fprs;
		if(small)
		{
			if(verbose)
				printf(  "   in ROC_average(): sample 100 points for every FPR decade (decades are automatically detected)\n");
			double fpr_min = 1, fpr_max = 0;
			for(int i=0; i<rocs.size(); i++)
			{
				for(int j=0; j<rocs[i].size(); j++)
				{
					if(rocs[i][j].fpr > fpr_max)
						fpr_max = rocs[i][j].fpr;
					else if(rocs[i][j].fpr && rocs[i][j].fpr < fpr_min)
						fpr_min = rocs[i][j].fpr;
				}
			}
			fprs = ucas::decades(fpr_min, fpr_max);
			if(verbose)
				printf(  "   in ROC_average(): %d decades detected, from %f to %f\n", int(fprs.size()), fprs.front(), fprs.back());
			fprs = ucas::subdivide(fprs, 100);
			if(verbose)
				printf(  "   in ROC_average(): %d FPR points\n", int(fprs.size()));
		}
		// compute all distinct FPR points from all ROCs
		else
		{
			if(verbose)
				printf(  "   in ROC_average(): compute all distinct FPR points from all ROCs\n");
			for(int i=0; i<rocs.size(); i++)
				for(int j=0; j<rocs[i].size(); j++)
					fprs.push_back(rocs[i][j].fpr);
			if(verbose)
				printf(  "   in ROC_average(): %d FPR points\n", int(fprs.size()));
			std::sort(fprs.begin(), fprs.end());
			fprs.erase(unique(fprs.begin(), fprs.end()), fprs.end());
			if(verbose)
				printf(  "   in ROC_average(): %d FPR points (after removing duplicates)\n", int(fprs.size()));
		}

		avg_roc.resize(fprs.size());
		for(size_t i=0; i<avg_roc.size(); i++)
		{
			avg_roc[i].fpr = fprs[i];
			avg_roc[i].P = rocs[0][0].P;
			avg_roc[i].N = rocs[0][0].N;
		}


		// distribute workload
		bool error = false;
		if(j > 1)
		{
			if(verbose)
				printf(  "   in ROC_average(): launch %d threads\n", j);
			std::vector <std::thread*> threads(j);
			ucas::Barrier barrier(j);
			for(int t=0; t<threads.size(); t++)
			{
				try
				{
					threads[t] = new std::thread(
						ROC_average_worker, std::ref(barrier), std::ref(rocs), std::ref(avg_roc), std::ref(error), verbose);
				}
				catch(...)
				{
					UCAS_THROW(ucas::strprintf("Failed to allocated thread %d (thread library not linked?)", t));
				}
			}

			// run and join
			for(int t=0; t<threads.size(); t++)
				threads[t]->join();

			// release memory
			for(int t=0; t<threads.size(); t++)
				delete threads[t];
		}
		else
		{
			ucas::Barrier barrier(1);
			if(verbose)
				printf(  "   in ROC_average(): run in single-thread mode\n");
			ROC_average_worker(barrier, rocs, avg_roc, error, verbose);
		}

		if(error)
			UCAS_THROW("in ROC_average(): error occurred");
	}

	// calculate max F1 from ROC curve
	template <typename T> 
	inline 
		size_t							// return index of ROC point with maximum F1
		F1_max(
		const std::vector< ROCpoint<T> > &ROC)// input ROC curve
		
	{
		// check preconditions
		if(ROC.empty())
			UCAS_THROW("in F1_max(): empty ROC");

		double f1_max = -inf<double>();
		size_t idx = 0;
		for(size_t i=0; i<ROC.size(); i++)
			if(ROC[i].F1() > f1_max)
			{
				f1_max = ROC[i].F1();
				idx = i;
			}
			return idx;
	}

	// calculate max MCC from ROC curve
	template <typename T> 
	inline 
		size_t							// return index of ROC point with maximum MCC
		MCC_max(
		const std::vector< ROCpoint<T> > &ROC)// input ROC curve
		
	{
		// check preconditions
		if(ROC.empty())
			UCAS_THROW("in MCC_max(): empty ROC");

		double mcc_max = -inf<double>();
		size_t idx = 0;
		for(size_t i=0; i<ROC.size(); i++)
			if(ROC[i].MCC() > mcc_max)
			{
				mcc_max = ROC[i].MCC();
				idx = i;
			}
			return idx;
	}

	// calculate max accuracy from ROC curve
	template <typename T> 
	inline 
		size_t							// return index of ROC point with maximum accuracy
		accuracy_max(
		const std::vector< ROCpoint<T> > &ROC)// input ROC curve
		
	{
		// check preconditions
		if(ROC.empty())
			UCAS_THROW("in accuracy_max(): empty ROC");

		double acc_max = -inf<double>();
		size_t idx = 0;
		for(size_t i=0; i<ROC.size(); i++)
			if(ROC[i].accuracy() > acc_max)
			{
				acc_max = ROC[i].accuracy();
				idx = i;
			}
			return idx;
	}


	// calculate AUC from ROC curve using trapezoidal rule
	template <typename T>
	inline 
		T											// return AUC
		AUC_trapz(
		const std::vector< ROCpoint<T> > & roc)		// ROC curve
		
	{
		// check preconditions
		if(roc.empty())
			UCAS_THROW("in AUC_trapz(): ROC is empty");
		if( (roc.front().tpr != 0 && roc.front().fpr != 0) && (roc.back().tpr != 1 && roc.back().fpr != 1) )
			UCAS_THROW("in AUC_trapz(): ROC not in the (TPR,FPR) range [(0,0),(1,1)]");

		// calculate AUC using trapezoidal rule
		T out = 0;
		for(int i=0; i<roc.size()-1; i++)
			out += ((roc[i].tpr + roc[i+1].tpr)*ucas::abs(roc[i].fpr - roc[i+1].fpr))/2.0;
		return out;
	}

	// calculate AUC from sample score arrays using trapezoidal rule applied on the calculated ROC curve
	template <typename T>
	inline 
		T								// return AUC
		AUC_trapz(
		const std::vector<T> &pos,		// positive sample score array
		const std::vector<T> &neg,		// negative sample score array
		int j = 1,						// number of concurrent threads
		bool pos_greater_than_neg = 1,	// 1(0) = the higher(lower) the sample score, the higher the probability of being positive
		bool exact = true)				// true: generate all ROC points; false: generate a subset of ROC points by using the minority class samples
		
	{
		std::vector< ROCpoint<T> > roc = ROC_compute(pos, neg, pos_greater_than_neg, exact ? std::vector<double>() : ROC_points(pos, neg, pos_greater_than_neg, false));
		return AUC_trapz<T>(roc);
	}


	// WMW AUC worker for multithreading 
	template <typename T>
	inline void
		AUC_wmw_worker(
		std::vector<T> &pos,				// positive sample scores array
		std::vector<T> &neg,				// negative sample scores array
		size_t n0, size_t n1,				// range [n0, n1) of negative samples to consider for AUC calculation	
		bool pos_greater_than_neg,			// 1(0) = the higher(lower) the sample score, the higher the probability of being positive
		double &sum)						// output
	{
		// apply WMW rule
		sum = 0;
		if(pos_greater_than_neg)
		{
			for(auto & p: pos)
				for(size_t i=n0; i<n1; i++)
					if(p > neg[i])
						sum += 1.0;
					else if(p == neg[i])
						sum += 0.5;
		}
		else
		{
			for(auto & p: pos)
				for(size_t i=n0; i<n1; i++)
					if(p < neg[i])
						sum += 1.0;
					else if(p == neg[i])
						sum += 0.5;
		}
	}


	// calculate AUC with Wilcoxon-Mann-Whitney from positive and negative sample score arrays
	template <typename T>
	inline 
		T								// return AUC
		AUC_wmw(
		std::vector<T> &pos,			// positive sample scores array
		std::vector<T> &neg,			// negative sample scores array
		int j = 1,						// number of concurrent threads
		bool pos_greater_than_neg = 1)	// 1(0) = the higher(lower) the sample score, the higher the probability of being positive
	{
		// discard nonfinite scores
		pos.erase(std::remove_if(pos.begin(), pos.end(), is_nan), pos.end());
		neg.erase(std::remove_if(neg.begin(), neg.end(), is_nan), neg.end());

		// apply WMW rule
		double res = 0;
		if(j > 1)
		{
			int n_threads = j;
			std::vector <std::thread*> threads;
			std::vector <double> sums(n_threads);
			size_t n_neg_i = neg.size()/n_threads;
			size_t neg_count = 0;
			for(int t=0; t<n_threads; t++)
			{
				std::thread *thr = new std::thread(AUC_wmw_worker<T>, std::ref(pos), std::ref(neg), neg_count, neg_count + (t<neg.size()%n_threads ? n_neg_i+1 : n_neg_i), pos_greater_than_neg, std::ref(sums[t]));
				if(!thr)
					UCAS_THROW(ucas::strprintf("Failed to allocated thread %d/%d)", t+1, n_threads));
				threads.push_back(thr);
				neg_count += (t<neg.size()%n_threads ? n_neg_i+1 : n_neg_i);
			}

			// run and join
			for(int t=0; t<n_threads; t++)
				threads[t]->join();

			// release memory
			for(int t=0; t<n_threads; t++)
				delete threads[t];

			// calculate res
			for(int t=0; t<n_threads; t++)
				res += sums[t];
		}
		else
			AUC_wmw_worker(pos, neg, 0, neg.size(), pos_greater_than_neg, res);

		return res/(static_cast<T>(pos.size())*static_cast<T>(neg.size()));
	}


	// calculate PAUC from ROC curve using trapezoidal rule
	template <typename T>
	inline 
		T											// return PAUC
		PAUC_trapz(
		const std::vector< ROCpoint<T> > & roc,		// ROC curve
		T fpr0, T fpr1)								// range [fpr0, fpr1]
		
	{
		// check preconditions
		if(roc.empty())
			UCAS_THROW("in PAUC_trapz(): ROC is empty");
		if( (roc.front().tpr != 0 && roc.front().fpr != 0) && (roc.back().tpr != 1 && roc.back().fpr != 1) )
			UCAS_THROW("in PAUC_trapz(): ROC not in the (TPR,FPR) range [(0,0),(1,1)]");
		if(fpr0 < 0)
			UCAS_THROW(ucas::strprintf("in PAUC_trapz(): fpr0(%f) cannot be < 0", fpr0));
		if(fpr1 > 1)
			UCAS_THROW(ucas::strprintf("in PAUC_trapz(): fpr1(%f) cannot be > 1", fpr0));
		if(fpr1 <= fpr0)
			UCAS_THROW(ucas::strprintf("in PAUC_trapz(): fpr1(%f) cannot be <= fpr0 (%f)", fpr1, fpr0));

		// calculate PAUC using trapezoidal rule
		T out = 0;
		for(int i=0; i<roc.size()-1; i++)
			if(roc[i].fpr >= fpr0 && roc[i+1].fpr <= fpr1)
				out += ((roc[i].tpr + roc[i+1].tpr)*ucas::abs(roc[i+1].fpr - roc[i].fpr))/2.0;
		return out;
	}

	// calculate logPAUC / mean sensitivity S (meanS)
	// as proposed in Hupse, R. and Karssemeijer, N., "Use of Normal Tissue Context in Computer-Aided
	// Detection of Masses in Mammograms", IEEE Transactions on Medical Imaging, 2009.
	template <typename T>
	inline 
		T											// return meanS
		meanS(
		const std::vector< ROCpoint<T> > & roc,		// ROC curve
		T fpr0, T fpr1)								// range [fpr0, fpr1]
		
	{
		// check preconditions
		if(roc.empty())
			UCAS_THROW("in meanS(): ROC is empty");
		if(fpr0 <= 0)
			UCAS_THROW("in meanS(): fpr0 cannot be <= 0");
		if(fpr1 > 1)
			UCAS_THROW("in meanS(): fpr1 cannot be > 1");
		if(fpr1 <= fpr0)
			UCAS_THROW("in meanS(): fpr1 cannot be <= fpr0");
		if( fpr0 < roc.front().fpr)
			UCAS_THROW(ucas::strprintf("in meanS(): fpr0 (%f) is outside the ROC starting fpr (%f)", fpr0, roc.front().fpr));
		if( fpr1 > roc.back().fpr)
			UCAS_THROW(ucas::strprintf("in meanS(): fpr1 (%f) is outside the ROC ending fpr (%f)", fpr1, roc.back().fpr));

		// sample 100 points for every decade
		std::vector< double> fprs = ucas::decades(fpr0, fpr1);
		fprs = ucas::subdivide(fprs, 10);
		double res = 0;
		for(size_t i=0; i<fprs.size()-1; i++)
			res += ( ROC_interp_tpr(roc, fprs[i]) + ROC_interp_tpr(roc, fprs[i+1])) * ucas::abs(fprs[i+1]-fprs[i]) / (2*fprs[i]);
		res /= std::log(fpr1)-std::log(fpr0);

		//printf("fpr=[%f, %f], res=%f\n", fpr0, fpr1, res);
		if(res > 1)
			UCAS_THROW("in meanS(): result > 1 (you should never see this)");
		return res;
	}

	// save ROC curve
	template <typename T>
	static inline void
		saveROC(
		std::vector< ROCpoint<T> > &ROC,	// input ROC curve
		const std::string & path,			// output ROC file
		bool with_header = true,			// whether to write header with additional info (AUC, accuracy, F1, etc.)
		bool tpr_stdev = false)				// whether to replace the 'threshold' column with 'tpr_stdev'
		
	{
		std::ofstream f(path);
		if(!f.is_open())
			UCAS_THROW(ucas::strprintf("Cannot write ROC file at \"%s\"", path.c_str()));
		if(with_header)
		{
			f << "# positive samples = " << ROC[0].P << "\n";
			f << "# negative samples = " << ROC[0].N << "\n";
			f << "# AUC = " << ucas::f2str(AUC_trapz(ROC)) << "\n";
			f << "# logPAUC(10^-6,1) = " << ucas::f2str(meanS(ROC, T(1e-6), T(1))) << "\n";
			f << "# logPAUC(10^-6,10^-1) = " << ucas::f2str(meanS(ROC, T(1e-6), T(1e-1))) << "\n";
			f << "# logPAUC(10^-6,10^-2) = " << ucas::f2str(meanS(ROC, T(1e-6), T(1e-2))) << "\n";
			f << "# logPAUC(10^-6,10^-3) = " << ucas::f2str(meanS(ROC, T(1e-6), T(1e-3))) << "\n";
			size_t acc_max_idx = accuracy_max(ROC);
			size_t f1_max_idx = F1_max(ROC);
			size_t mcc_max_idx = MCC_max(ROC);
			f << "# accuracy max = " << ucas::f2str(ROC[acc_max_idx].accuracy()) << " (for threshold = " << ucas::f2str(ROC[acc_max_idx].t) << ")\n";
			f << "# F1-score max = " << ucas::f2str(ROC[f1_max_idx].F1()) << " (for threshold = " << ucas::f2str(ROC[f1_max_idx].t) << ")\n";
			f << "# MCC      max = " << ucas::f2str(ROC[mcc_max_idx].MCC()) << " (for threshold = " << ucas::f2str(ROC[mcc_max_idx].t) << ")\n";
			f << "#\n";
			f << "# " << (tpr_stdev ? "stdev(TPR)" : "decision threshold") << "\tTPR\tFPR\n";
		}
		for(auto & p : ROC)
			f << ucas::f2str(tpr_stdev ? p.tpr_std : p.t).c_str() << "\t" << ucas::f2str(p.tpr) << "\t" << ucas::f2str(p.fpr) << "\n";
		f.close();
	}

	// save positive / negative sample probability scores
	template<typename T>
	static inline void
		save_sco(
		const std::vector<T> & pos_scores, 
		const std::vector<T> & neg_scores,
		const std::string & pos_sco_file,
		const std::string & neg_sco_file,
		bool append_sco,
		T *pos_mean = 0,
		T *neg_mean = 0)
	
	{
		if(pos_mean && neg_mean)
		{
			*pos_mean=0;
			*neg_mean=0;
			for(int i=0; i<pos_scores.size(); i++)
				*pos_mean += pos_scores[i];
			for(int i=0; i<neg_scores.size(); i++)
				*neg_mean += neg_scores[i];
			*pos_mean /= pos_scores.size();
			*neg_mean /= neg_scores.size();
		}
		if(pos_scores.empty() && neg_scores.empty())
			UCAS_THROW("No probability scores found");


		// Save probability score files
		for(int s=0; s<2; s++)
		{
			bool filexists = ucas::isFile(s ? pos_sco_file :  neg_sco_file);
			std::ofstream f(s ? pos_sco_file :  neg_sco_file, append_sco ? std::ios_base::app : std::ios_base::trunc);
			if(!f.is_open())
				throw ucas::CannotOpenFileError(s ? pos_sco_file :  neg_sco_file);
			if(!append_sco || (append_sco && !filexists))
				f << "#SAMPLE \t#SCORE\n";
			for(size_t k=0; k < (s ? pos_scores.size() : neg_scores.size()); k++)
				f << ucas::strprintf("%8d\t%f\n", k, s ? pos_scores[k] : neg_scores[k]);
			f.close();
		}
	}

	struct FROCpoint
	{
		float threshold, TPR, FPR;
		FROCpoint() : threshold(0.0f), TPR(0.0f), FPR(0.0f){}
		FROCpoint(float t, float tpr, float fpr) : threshold(t), TPR(tpr), FPR(fpr){}
	};


	// METRICs
	template <typename T>
	class BinaryClassMetric
	{
		public:

			// metric name
			virtual std::string name(bool with_parameters = true) = 0;

			// metric unique id (includes parameter values, if any)
			virtual std::string id() = 0;

			// do higher values means better performance?
			virtual bool higherIsBetter() = 0;

			// is a ROC-based metric ?
			virtual bool isROCMetric() = 0;

			// compute metric from the given binary class samples distributions
			virtual T evalFromSamples(
				const std::vector <T> & pos,		// positive sample scores
				const std::vector <T> & neg,		// negative sample scores
				bool pos_greater_than_neg = true,	// 1(0) = the higher(lower) the sample score, the higher the probability of being positive
				T threshold = 0)					// positive / negative threshold
				 = 0;

			// compute metric from the given ROC
			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC )  = 0;

			// factory method
			static BinaryClassMetric<T>* instance(const std::string & str) ;

			// list of available metrics
			static std::string availableMetrics();
	};

	template <typename T>
	class AUC_Metric : public BinaryClassMetric<T>
	{
		public:

			virtual std::string name(bool with_parameters = true)		{ return "AUC"; }

			virtual std::string id()		{ return "AUC"; }

			bool isROCMetric()				{ return true;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				return ucas::AUC_trapz<T>(pos, neg, 1, pos_greater_than_neg);
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				return ucas::AUC_trapz<T>(ROC);
			}
	};


	template <typename T>
	class logPAUC_Metric : public BinaryClassMetric<T>
	{
		private:

			T fpr0, fpr1;

		public:

			logPAUC_Metric() : fpr0(1e-6), fpr1(1){}

			logPAUC_Metric(const std::string & str) 
			{
				std::string args = str.substr(str.find_first_of("(")+1);
				args = args.substr(0, args.find_last_of(")"));

				std::vector<std::string> tokens;
				ucas::split(args, ",", tokens);

				if(tokens.size() != 2)
					UCAS_THROW("expected 2 fprs separated by \",\" (comma)");

				fpr0 = str2num<T>(tokens[0]);
				fpr1 = str2num<T>(tokens[1]);

				if(fpr0 < 0)
					UCAS_THROW("fpr0 cannot be < 0");
				if(fpr1 > 1)
					UCAS_THROW("fpr1 cannot be > 1");
				if(fpr0 >= fpr1)
					UCAS_THROW(ucas::strprintf("fpr0(%f) cannot be >=  fpr1 (%f)", fpr0, fpr1));
			}

			virtual std::string name(bool with_parameters = true)		{ return with_parameters ? "logPAUC(fpr0,fpr1)" : "logPAUC"; }

			virtual std::string id()		{ return ucas::strprintf("logPAUC(%f,%f)", fpr0, fpr1); }

			bool isROCMetric()				{ return true;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				std::vector < ucas::ROCpoint<T> > ROC = ucas::ROC_compute<T>(pos, neg, pos_greater_than_neg);
				return ucas::meanS<T>(ROC, fpr0, fpr1);
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				return ucas::meanS<T>(ROC, fpr0, fpr1);
			}
	};


	template <typename T>
	class PAUC_Metric : public BinaryClassMetric<T>
	{
		private:

			T fpr0, fpr1;

		public:

			PAUC_Metric() : fpr0(0), fpr1(1){}

			PAUC_Metric(const std::string & str) 
			{
				std::string args = str.substr(str.find_first_of("(")+1);
				args = args.substr(0, args.find_last_of(")"));

				std::vector<std::string> tokens;
				ucas::split(args, ",", tokens);

				if(tokens.size() != 2)
					UCAS_THROW("expected 2 fprs separated by \",\" (comma)");

				fpr0 = str2num<T>(tokens[0]);
				fpr1 = str2num<T>(tokens[1]);

				if(fpr0 < 0)
					UCAS_THROW("fpr0 cannot be < 0");
				if(fpr1 > 1)
					UCAS_THROW("fpr1 cannot be > 1");
				if(fpr0 >= fpr1)
					UCAS_THROW(ucas::strprintf("fpr0(%f) cannot be >=  fpr1 (%f)", fpr0, fpr1));
			}

			virtual std::string name(bool with_parameters = true)		{ return with_parameters ? "PAUC(fpr0,fpr1)" : "PAUC"; }

			virtual std::string id()		{ return ucas::strprintf("PAUC(%f,%f)", fpr0, fpr1); }

			bool isROCMetric()				{ return true;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				std::vector < ucas::ROCpoint<T> > ROC = ucas::ROC_compute<T>(pos, neg, pos_greater_than_neg);
				return ucas::PAUC_trapz<T>(ROC, fpr0, fpr1);
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				return ucas::PAUC_trapz<T>(ROC, fpr0, fpr1);
			}
	};


	template <typename T>
	class TPR_Metric : public BinaryClassMetric<T>
	{
		public:

			virtual std::string name(bool with_parameters = true)		{ return "TPR"; }

			virtual std::string id()		{ return "TPR"; }

			bool isROCMetric()				{ return false;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				T TP=0, TN=0, FP=0, FN=0;
				eval(pos, neg, TP, FP, TN, FN, pos_greater_than_neg, threshold);
				return TP/pos.size();
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				UCAS_THROW("evaluating from ROC is meaningless: you should never see this");
			}
		};


		template <typename T>
		class FPR_Metric : public BinaryClassMetric<T>
		{
		public:

			virtual std::string name(bool with_parameters = true)		{ return "FPR"; }

			virtual std::string id()		{ return "FPR"; }

			bool isROCMetric()				{ return false;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				T TP=0, TN=0, FP=0, FN=0;
				eval(pos, neg, TP, FP, TN, FN, pos_greater_than_neg, threshold);
				return FP/neg.size();
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				UCAS_THROW("evaluating from ROC is meaningless: you should never see this");
			}
	};


	template <typename T>
	class Precision_Metric : public BinaryClassMetric<T>
	{
		public:

			virtual std::string name(bool with_parameters = true)		{ return "precision"; }

			virtual std::string id()		{ return "precision"; }

			bool isROCMetric()				{ return false;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				T TP=0, TN=0, FP=0, FN=0;
				eval(pos, neg, TP, FP, TN, FN, pos_greater_than_neg, threshold);
				return TP / (TP + FP);
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				UCAS_THROW("evaluating from ROC is meaningless: you should never see this");
			}
	};


	template <typename T>
	class Accuracy_Metric : public BinaryClassMetric<T>
	{
		public:

			virtual std::string name(bool with_parameters = true)		{ return "accuracy"; }

			virtual std::string id()		{ return "accuracy"; }

			bool isROCMetric()				{ return false;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				T TP=0, TN=0, FP=0, FN=0;
				eval(pos, neg, TP, FP, TN, FN, pos_greater_than_neg, threshold);
				return (TP + TN) / (TP + TN + FP + FN);
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				UCAS_THROW("evaluating from ROC is meaningless: you should never see this");
			}
	};


	template <typename T>
	class F1_Metric : public BinaryClassMetric<T>
	{
		public:

			virtual std::string name(bool with_parameters = true)		{ return "F1"; }

			virtual std::string id()		{ return "F1"; }

			bool isROCMetric()				{ return false;  }

			virtual bool higherIsBetter()	{ return true;  }

			virtual T evalFromSamples(const std::vector <T> & pos, const std::vector <T> & neg, bool pos_greater_than_neg = true, T threshold = 0) 
			{
				T TP=0, TN=0, FP=0, FN=0;
				eval(pos, neg, TP, FP, TN, FN, pos_greater_than_neg, threshold);
				double precision = TP / (TP + FP);
				double recall = TP / (TP + FN);
				return 2 * (precision * recall) / (precision + recall);
			}

			virtual T evalFromROC( const std::vector < ROCpoint<T> > & ROC ) 
			{
				UCAS_THROW("evaluating from ROC is meaningless: you should never see this");
			}
	};


	template <typename T>
	BinaryClassMetric<T>* BinaryClassMetric<T>::instance(const std::string & str) 
	{
		if(str == AUC_Metric<T>().name())
			return new AUC_Metric<T>();
		else if(str.find(logPAUC_Metric<T>().name(false)) != std::string::npos)
			return new logPAUC_Metric<T>(str);
		else if(str.find(PAUC_Metric<T>().name(false)) != std::string::npos)
			return new PAUC_Metric<T>(str);
		else if(str == TPR_Metric<T>().name())
			return new TPR_Metric<T>();
		else if(str == FPR_Metric<T>().name())
			return new FPR_Metric<T>();
		else if(str == Accuracy_Metric<T>().name())
			return new Accuracy_Metric<T>();
		else if(str == Precision_Metric<T>().name())
			return new Precision_Metric<T>();
		else if(str == F1_Metric<T>().name())
			return new F1_Metric<T>();
		else
			UCAS_THROW(ucas::strprintf("Cannot recognize metric \"%s\"", str.c_str()));
	}

	template <typename T>
	std::string BinaryClassMetric<T>::availableMetrics()
	{
		return AUC_Metric<T>().name() + "," + logPAUC_Metric<T>().name() + "," + PAUC_Metric<T>().name() 
			+ "," + TPR_Metric<T>().name() + "," + FPR_Metric<T>().name() + "," + Precision_Metric<T>().name()
			+ "," + Accuracy_Metric<T>().name() + "," + F1_Metric<T>().name();
	}
}
