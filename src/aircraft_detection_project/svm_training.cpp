#include "svm_training.h"

#include <random>
#include <filesystem>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "hog_features_extraction.h"

namespace fs = std::filesystem;

// Function declarations
bool processImage(const std::string& image_file_path, const std::string& annotation_file_path, int image_width, int image_height, int num_false_positives_per_img, std::vector<std::vector<float>>& true_positive_hog_features, std::vector<std::vector<float>>& false_positive_hog_features);

void generateFalsePositiveBoxes(int num_false_positives, int image_width, int image_height, const std::vector<cv::Rect>& true_positive_boxes, std::vector<cv::Rect>& false_positive_boxes);

void generateNearbyFalsePositives(int numNearFP, int image_width, int image_height, const std::vector<cv::Rect>& true_positive_boxes, std::vector<cv::Rect>& false_positive_boxes, std::mt19937& gen, std::uniform_int_distribution<>& disOffset, std::uniform_int_distribution<>& disSize);

void generateFarFalsePositives(int numFarFP, int image_width, int image_height, const std::vector<cv::Rect>& true_positive_boxes, std::vector<cv::Rect>& false_positive_boxes, std::mt19937& gen, std::uniform_int_distribution<>& disPosX, std::uniform_int_distribution<>& disPosY, std::uniform_int_distribution<>& disSize);

bool is_significantly_overlapping(const cv::Rect& box1, const cv::Rect& box2, double threshold = 0.2);




// Extracts HOG features from images in the training dataset and saves them to CSV files for SVM training
void extract_csv_for_svm_training()
{
    constexpr int image_width = 4800;
    constexpr int image_height = 2703;
    constexpr int num_false_positives_per_img = 24;

    // Containers for HOG features of true positives and false positives
    std::vector<std::vector<float>> true_positive_hog_features;
    std::vector<std::vector<float>> false_positive_hog_features;

    // Iterate over all files in the training dataset directory
    for (const auto& entry : fs::directory_iterator(TRAINING_DATASET_PATH))
    {
        const std::string filename = entry.path().string();

        // Process only .txt files (YOLO annotations)
        if (filename.find(".txt") != std::string::npos)
        {
            const std::string base_name = filename.substr(0, filename.find_last_of('.'));
            const std::string image_file_path = base_name + ".jpg";

            // Check if the corresponding image file exists
            if (!processImage(image_file_path, filename, image_width, image_height, num_false_positives_per_img, true_positive_hog_features, false_positive_hog_features))
                std::cerr << "Failed to process image and annotations for: " << image_file_path << "\n";
        }
    }

    // Create output directory for SVM training input
    const std::filesystem::path output_dir = createDirectory(std::filesystem::path(SRC_DIR_PATH), "svm_training_input");

    // Define paths for output CSV files
    const std::filesystem::path tp_file_path = output_dir / "tp_training.csv";
    const std::filesystem::path fp_file_path = output_dir / "fp_training.csv";

    // Write HOG features to CSV files
    try
    {
        writeHogFeaturesToCsv(true_positive_hog_features, tp_file_path.string());
        writeHogFeaturesToCsv(false_positive_hog_features, fp_file_path.string());
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error writing to CSV file: " << e.what() << "\n";
    }
}


// Processes a single image and its corresponding YOLO annotation file
bool processImage(const std::string& image_file_path, const std::string& annotation_file_path, int image_width, int image_height, int num_false_positives_per_img, std::vector<std::vector<float>>& true_positive_hog_features, std::vector<std::vector<float>>& false_positive_hog_features)
{
    // Check if the image file exists
    if (!fs::exists(image_file_path)) 
    {
        std::cerr << "Error: Image file does not exist: " << image_file_path << "\n";
        return false;
    }

    // Load the image in grayscale
    cv::Mat image = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);
    if (!image.data) 
    {
        std::cerr << "Error: Could not open or find the image: " << image_file_path << "\n";
        return false;
    }

    // Read the YOLO annotation boxes for true positives
    std::vector<cv::Rect> true_positive_boxes;
    try
    {
        true_positive_boxes = readYoloBoxes(annotation_file_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return false;
    }

    // Generate false positive boxes
    std::vector<cv::Rect> false_positive_boxes;
    generateFalsePositiveBoxes(num_false_positives_per_img, image_width, image_height, true_positive_boxes, false_positive_boxes);

    // Extract HOG features for true positives and false positives
    auto tp_features = hog_features_extraction(true_positive_boxes, image);
    auto fp_features = hog_features_extraction(false_positive_boxes, image);

    // Add extracted features to the respective containers
    true_positive_hog_features.insert(true_positive_hog_features.end(), tp_features.begin(), tp_features.end());
    false_positive_hog_features.insert(false_positive_hog_features.end(), fp_features.begin(), fp_features.end());

    return true;
}

// Generates false positive bounding boxes for an image
void generateFalsePositiveBoxes(int num_false_positives, int image_width, int image_height, const std::vector<cv::Rect>& true_positive_boxes, std::vector<cv::Rect>& false_positive_boxes)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> disOffset(-50, 50);          // Small perturbation around true positives
    std::uniform_int_distribution<> disPosX(0, image_width - 1); // Random coordinates for far false positives
    std::uniform_int_distribution<> disPosY(0, image_height - 1);
    std::uniform_int_distribution<> disSize(0, roi_sizes.size() - 1);

    const int numNearFP = num_false_positives / 2;        // Half of the false positives are near true positives
    const int numFarFP = num_false_positives - numNearFP; // Half of the false positives are far from true positives

    generateNearbyFalsePositives(numNearFP, image_width, image_height, true_positive_boxes, false_positive_boxes, gen, disOffset, disSize);
    generateFarFalsePositives(numFarFP, image_width, image_height, true_positive_boxes, false_positive_boxes, gen, disPosX, disPosY, disSize);
}

// Generates nearby false positive bounding boxes (close to true positives)
void generateNearbyFalsePositives(int numNearFP, int image_width, int image_height, const std::vector<cv::Rect>& true_positive_boxes, std::vector<cv::Rect>& false_positive_boxes, std::mt19937& gen, std::uniform_int_distribution<>& disOffset, std::uniform_int_distribution<>& disSize)
{
    int generatedNearFP = 0;
    for (const auto& tp : true_positive_boxes)
    {
        if (generatedNearFP >= numNearFP) break; // Ensure not to generate more false positives than necessary

        int dx = disOffset(gen);
        int dy = disOffset(gen);
        cv::Point fp(tp.x + tp.width / 2 + dx, tp.y + tp.height / 2 + dy); // Perturbed center point

        // Ensure the false positive point is within image boundaries
        if (fp.x >= 0 && fp.y >= 0 && fp.x < image_width && fp.y < image_height) 
        {
            bool tooMuchIntersection = false;
            for (const auto& box : true_positive_boxes)
            {
                if (is_significantly_overlapping(cv::Rect(fp.x, fp.y, 1, 1), box, 0.2))
                {
                    tooMuchIntersection = true;
                    break;
                }
            }
            if (!tooMuchIntersection) // Ensure the false positive point is not too close to any true positive
            {
                cv::Size roiSize = roi_sizes[disSize(gen)]; // Random size for the false positive ROI
                // Create the false positive bounding box
                cv::Rect false_positive_box(fp.x - roiSize.width / 2, fp.y - roiSize.height / 2, roiSize.width, roiSize.height);
                // Ensure the generated box is within image boundaries
                if (false_positive_box.x >= 0 && false_positive_box.y >= 0 && false_positive_box.x + false_positive_box.width <= image_width && false_positive_box.y + false_positive_box.height <= image_height)
                {
                    false_positive_boxes.push_back(false_positive_box);
                    generatedNearFP++;
                }
            }
        }
    }
}


// Generates far false positive bounding boxes (far from true positives)
void generateFarFalsePositives(int numFarFP, int image_width, int image_height, const std::vector<cv::Rect>& true_positive_boxes, std::vector<cv::Rect>& false_positive_boxes, std::mt19937& gen, std::uniform_int_distribution<>& disPosX, std::uniform_int_distribution<>& disPosY, std::uniform_int_distribution<>& disSize)
{
    int generatedFarFP = 0;
    while (generatedFarFP < numFarFP)
    {
        int x = disPosX(gen);
        int y = disPosY(gen);
        cv::Size roiSize = roi_sizes[disSize(gen)];
        cv::Rect false_positive_box(x - roiSize.width / 2, y - roiSize.height / 2, roiSize.width, roiSize.height);
        // Ensure the generated box is within image boundaries
        if (false_positive_box.x >= 0 && false_positive_box.y >= 0 && false_positive_box.x + false_positive_box.width <= image_width && false_positive_box.y + false_positive_box.height <= image_height) 
        {
            bool overlapping = false;
            for (const auto& tp_box : true_positive_boxes)
            {
                if (is_significantly_overlapping(false_positive_box, tp_box, 0.2))
                {
                    overlapping = true;
                    break;
                }
            }
            if (!overlapping)
            {
                false_positive_boxes.push_back(false_positive_box);
                generatedFarFP++;
            }
        }
    }
}

// Helper function to check if two rectangles significantly overlap
bool is_significantly_overlapping(const cv::Rect& box1, const cv::Rect& box2, double threshold)
{
    cv::Rect intersection = box1 & box2;
    double intersection_area = intersection.area();
    double box2_area = box2.area();

    return (intersection_area / box2_area) > threshold;
}