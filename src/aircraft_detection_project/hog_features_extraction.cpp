#include "hog_features_extraction.h"
#include "utils.h"
#include <iomanip>




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
        cv::resize(roi_img, resized_roi_img, cv::Size(64, 64)); // use INTER_AREA for shrinking

        // Compute the HOG descriptors for the resized ROI
        std::vector<float> descriptors;
        hog.compute(resized_roi_img, descriptors);

        // Append the HOG descriptors to the result
        hog_features.emplace_back(std::move(descriptors));
    }
    return hog_features;
}



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