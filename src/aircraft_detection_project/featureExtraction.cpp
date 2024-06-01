#include "featureExtraction.h"

// Funzione per calcolare le HOG features
std::vector<float> hogFeatures(const cv::Mat &roi) {
    int n_bins = 9; // Numero di bins per l'istogramma del gradiente
    int cell_size = 8;
    int block_size = 5 * cell_size;
    cv::Mat resized_roi;
    cv::resize(roi, resized_roi, cv::Size(256, 512));

    cv::HOGDescriptor hog(
        cv::Size(resized_roi.cols, resized_roi.rows),
        cv::Size(block_size, block_size),
        cv::Size(cell_size, cell_size),
        cv::Size(cell_size, cell_size), n_bins);

    std::vector<float> hog_features;
    hog.compute(resized_roi, hog_features);
    return hog_features;
}


// Funzione principale per estrarre le features e salvarle in file CSV
void featureExtractions(const std::map<std::string, std::vector<cv::Mat>> &samples_for_svm, const std::string &output_directory) {
    // Creazione dei percorsi per i file CSV
    std::filesystem::path dir_path(output_directory);
    std::filesystem::path file_path_fp = dir_path / "features_fp.csv";
    std::filesystem::path file_path_tp = dir_path / "features_tp.csv";

    // Apertura dei file CSV
    std::ofstream ofs_fp(file_path_fp);
    std::ofstream ofs_tp(file_path_tp);

    if (!ofs_fp.is_open() || !ofs_tp.is_open()) {
        std::cerr << "Errore nell'apertura dei file CSV per scrittura." << std::endl;
        return;
    }

    //itero sugli elementi della map attraverso un for con structured binding
    for (const auto &[key, samples] : samples_for_svm) {
        std::ofstream &ofs = (key == "FP") ? ofs_fp : ofs_tp;
        for (const auto &sample : samples) {
            std::vector<float> hog_features = hogFeatures(sample);
			    for (const auto &feature : hog_features) {
        			ofs << "," << feature;
				}
				ofs << "\n";
					}
    }

    ofs_fp.close();
    ofs_tp.close();
}