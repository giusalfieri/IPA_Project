#include "hog_features_extraction.h"
#include "utils.h"
#include <iomanip>



/**
 * @brief Extracts HOG features from specified regions of interest (ROIs) in an image.
 *
 * This function takes a vector of regions of interest (ROIs) and an image, extracts
 * each ROI from the image, resizes it to 64x64, and computes the HOG (Histogram of
 * Oriented Gradients) descriptors for each resized ROI. The HOG features are then
 * returned in a vector of vectors, where each inner vector corresponds to the HOG
 * descriptors of a single ROI.
 *
 * @param[in] rois A vector of `cv::Rect` defining the regions of interest in the image.
 * @param[in] image The input image from which the ROIs are extracted.
 * @return A vector of vectors, where each inner vector contains the HOG descriptors
 *         for a corresponding ROI.
 *
 * @see cv::HOGDescriptor
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
 * This function takes a vector of HOG feature vectors and writes them to a specified
 * CSV file. Each row in the CSV file corresponds to one HOG feature vector, and each
 * value in the vector is written with a fixed precision of 6 decimal places.
 *
 * @param[in] hog_features A vector of HOG feature vectors to be written to the CSV file.
 * @param[in] filename The name of the CSV file to write the HOG features to.
 *
 * @note The file is opened using the `openFile` function, which is assumed to return a
 *       file stream. The features are written in a consistent format with a fixed
 *       precision to ensure proper formatting regardless of locale settings.
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