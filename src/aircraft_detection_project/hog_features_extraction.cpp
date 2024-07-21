#include "hog_features_extraction.h"
#include "utils.h"
#include <iomanip>



/**
 * @brief Extracts Histogram of Oriented Gradients (HOG) features from regions of interest (ROIs) in an image.
 *
 * This function takes a vector of regions of interest (ROIs) and an image, extracts each ROI from the image,
 * resizes it to 64x64 pixels, and then computes the HOG descriptors for each resized ROI. The HOG features
 * for all ROIs are returned as a vector of vectors of floats.
 *
 * @param rois A vector of cv::Rect objects representing the regions of interest in the image.
 * @param image A cv::Mat object representing the input image from which the ROIs are extracted.
 * @return std::vector<std::vector<float>> A vector of vectors, where each inner vector contains the HOG descriptors for a corresponding ROI.
 */
std::vector< std::vector<float> > hog_features_extraction(const std::vector<cv::Rect>& rois, const cv::Mat& image)
{
    // Create a HOG descriptor object
    cv::HOGDescriptor hog(cv::Size(64, 64),
        cv::Size(8, 8),
        cv::Size(8, 8),
        cv::Size(8, 8), 9);

    // Avoids multiple reallocations by reserving space for the HOG features of all ROIs
	std::vector<std::vector<float>> hog_features;
    hog_features.reserve(rois.size());

    for (const auto& roi : rois)
    {
        // Extract the region of interest from the image
        cv::Mat roi_img = image(roi);

        // Resize the ROI to 64x64
        cv::Mat resized_roi_img;
        cv::resize(roi_img, resized_roi_img, cv::Size(64, 64), 0, 0, cv::INTER_AREA);


        // Compute the HOG descriptors for the resized ROI
        std::vector<float> descriptors;
        hog.compute(resized_roi_img, descriptors);

        // Append the HOG descriptors to the result
        hog_features.emplace_back(std::move(descriptors));
    }
    return hog_features;
}


/**
 * @brief Writes HOG features to a CSV file.
 *
 * This function takes a vector of HOG features and writes them to a specified CSV file. Each set of HOG features
 * is written as a single line in the file, with individual features separated by commas. The features are written
 * with fixed precision and 6 decimal places to ensure a consistent format.
 *
 * @param hog_features A vector of vectors, where each inner vector contains the HOG descriptors for a particular ROI.
 * @param filename The name of the CSV file to which the HOG features will be written.
 */
void writeHogFeaturesToCsv(const std::vector<std::vector<float>>& hog_features, const std::string& filename)
{
    auto file = openFile(filename);

    for (const auto& features : hog_features)
    {
        for (auto it = features.cbegin(); it != features.cend(); ++it)
        {
            // Add a comma before each feature except the first one
            if (it != features.cbegin())
                file << ",";

            // Write the feature to the file with fixed precision and 6 decimal places (e.g., 0.123456)
            // This is done to ensure that the features are written in a consistent format, regardless of the locale settings
            // (e.g., using a comma as the decimal separator in some locales)
            file << std::fixed << std::setprecision(6) << *it;
        }
        file << "\n";
    }
}