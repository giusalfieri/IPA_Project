#include "svm_training.h"

#include "utils.h"
#include "hog_features_extraction.h"
#include "template_matching.h"
#include <random>
#include <filesystem>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>




//==============================================================================
//                                      Forward Declarations
void classifyPointsByYoloBoxes(const std::vector<cv::Rect>& yolo_boxes, const std::vector<cv::Point>& max_corr_points, std::vector<cv::Point>& max_corr_points_inside_yolo, std::vector<cv::Point>& max_corr_points_outside_yolo);

std::vector<cv::Rect> selectROIsWithHighestIoU(const std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>>& yoloBox_roi_pairs);

std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> associateYoloBoxesWithRois(const std::vector<cv::Rect>& yolo_boxes, const std::vector<cv::Point>& max_corr_tp);
//==============================================================================




// ============================================================================= 
// =============================================================================
//                               SVM TRAINING 
// =============================================================================
// =============================================================================

/**
 * @brief Generates SVM training data by extracting HOG features and saving them to CSV files.
 *
 * This function processes a dataset of images and their corresponding YOLO labels to generate training data for an SVM.
 * It performs template matching, classifies points based on their location relative to YOLO bounding boxes,
 * extracts HOG features for true positives and false positives, and saves the features to CSV files.
 *
 * The function performs the following steps:
 * 1. Lists directories for k-means clustering by size and calculates average dimensions for ROIs.
 * 2. Reads dataset image paths and YOLO label paths.
 * 3. Reads images in grayscale.
 * 4. Iterates through each image in the dataset:
 *    a. Performs template matching.
 *    b. Reads YOLO bounding boxes.
 *    c. Classifies points inside and outside YOLO boxes.
 *    d. Filters points outside YOLO boxes by minimum distance.
 *    e. Associates each YOLO box with ROIs extracted from points inside it.
 *    f. Selects ROIs with the highest Intersection over Union (IoU) for true positives.
 *    g. Extracts ROIs for false positives.
 *    h. Extracts HOG features for true positives and false positives.
 *    i. Stores the HOG features in vectors, reserving space to minimize reallocations.
 * 5. Saves the HOG features to CSV files for SVM training.
 *
 * @note The function assumes that the dataset images and YOLO label files are in the specified directory.
 * @note The function initializes a random number generator for choosing random ROI sizes to extract false positives.
 *
 * @see listDirectories
 * @see calculateAvgDims
 * @see globFiles
 * @see readImages
 * @see templateMatching
 * @see readYoloBoxes
 * @see classifyPointsByYoloBoxes
 * @see filterPointsByMinDistance
 * @see associateYoloBoxesWithRois
 * @see selectROIsWithHighestIoU
 * @see hog_features_extraction
 * @see writeHogFeaturesToCsv
 */
void generateSvmTrainingData()
{
   
    std::vector<std::string> kmeans_by_size_clusters;
    listDirectories(std::filesystem::path(SRC_DIR_PATH) / "kmeans_by_size", kmeans_by_size_clusters);


    std::vector<cv::Size> roi_sizes;
    roi_sizes.reserve(kmeans_by_size_clusters.size());
    for (size_t i = 0; i < kmeans_by_size_clusters.size(); i++)
        roi_sizes.push_back(calculateAvgDims(std::filesystem::path(kmeans_by_size_clusters[i])));


    // Initialize random number generator for choosing random ROI sizes to extract false positives
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, roi_sizes.size() - 1); // Uniform distribution between 0 and roi_sizes.size() - 1


    
    std::vector<std::string> dataset_img_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.jpg", dataset_img_paths);
    // Read yolo labels paths for the dataset images
    std::vector<std::string> yolo_labels_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.txt", yolo_labels_paths);


    // Read images in grayscale
    std::vector<cv::Mat> src_imgs_gray;
    readImages(dataset_img_paths, src_imgs_gray, cv::IMREAD_GRAYSCALE);


    // These vectors will contain the HOG features for the true positives and false positives of all training images
    std::vector<std::vector<float>> true_positive_hog_features;
    std::vector<std::vector<float>> false_positive_hog_features;

    const auto dataset_training_cardinality = dataset_img_paths.size();
    const auto buffer_size = 100;
   
    
    for (size_t i=0;i< dataset_training_cardinality;i++)
	{
        // Perform template matching 
        std::vector<cv::Point> matched_points = templateMatching(src_imgs_gray[i]);

        // Read YOLO bounding boxes for the current image
        std::vector<cv::Rect> yolo_boxes = readYoloBoxes(yolo_labels_paths[i], src_imgs_gray[i]);

        // Classify points by their position inside or outside YOLO boxes
        std::vector<cv::Point> max_corr_points_inside_yolo;
    	std::vector<cv::Point> max_corr_points_outside_yolo;
        classifyPointsByYoloBoxes(yolo_boxes, matched_points, max_corr_points_inside_yolo, max_corr_points_outside_yolo);

        // Filter outside yolo points by minimum distance between them
        std::vector<cv::Point> max_corr_points_out_yolo = filterPointsByMinDistance(max_corr_points_outside_yolo, 100);

    	// Associate each YOLO box with the ROIs extracted from the points inside it
        // The result is a vector of pairs, where each pair contains a YOLO box and the ROIs associated with it
        std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> yoloBox_roi_pairs = associateYoloBoxesWithRois(yolo_boxes, max_corr_points_inside_yolo);

        
        std::vector<cv::Rect> tp_rois = selectROIsWithHighestIoU(yoloBox_roi_pairs);
        
        // FP ROI extraction 
        std::vector<cv::Rect> fp_rois;

        for (const auto& point : max_corr_points_out_yolo)
        {
            // Choose a random ROI size between the available sizes in roi_sizes
            const auto& roi_size = roi_sizes[dis(gen)];

            const int x = point.x - roi_size.width / 2;
            const int y = point.y - roi_size.height / 2;

            if (cv::Rect roi(x, y, roi_size.width, roi_size.height); isRoiInImage(roi))
            {
                bool overlapping = false;

                // Check for overlap with tp_rois
                for (const auto& tp_box : tp_rois)
                {
                    if ((tp_box & roi).area() != 0)
                    {
                        overlapping = true;
                        break;
                    }
                }

                // Check for overlap with yolo_boxes if no overlap with tp_rois
                if (!overlapping)
                {
                    for (const auto& yolo_box : yolo_boxes)
                    {
                        if ((yolo_box & roi).area() != 0)
                        {
                            overlapping = true;
                            break;
                        }
                    }
                }

                if (!overlapping)
                    fp_rois.push_back(roi);
            }
        }

        std::vector<std::vector<float>> new_tp_hog_features = hog_features_extraction(tp_rois, src_imgs_gray[i]);
        std::vector<std::vector<float>> new_fp_hog_features = hog_features_extraction(fp_rois, src_imgs_gray[i]);

        // N.B buffer_size is an arbitrary buffer chosen to reduce the number of reallocations
    	// needed when inserting new elements into the vectors. The idea is to reserve 
    	// slightly more space than necessary to minimize the number of times the vector 
    	// needs to reallocate memory, which is a costly operation.
        if (true_positive_hog_features.capacity() < true_positive_hog_features.size() + new_tp_hog_features.size())
        {
            true_positive_hog_features.reserve(true_positive_hog_features.size() + new_tp_hog_features.size() + buffer_size);
        }
        if (false_positive_hog_features.capacity() < false_positive_hog_features.size() + new_fp_hog_features.size())
        {
            false_positive_hog_features.reserve(false_positive_hog_features.size() + new_fp_hog_features.size() + buffer_size);
        }

        true_positive_hog_features.insert(true_positive_hog_features.end(), new_tp_hog_features.begin(), new_tp_hog_features.end());
        false_positive_hog_features.insert(false_positive_hog_features.end(), new_fp_hog_features.begin(), new_fp_hog_features.end());
	}


    //-------------------- SAVING THE HOG FEATURES TO CSV FILES REQUIRED FOR SVM TRAINING ----------------------


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


/**
 * @brief Classifies points based on whether they fall inside or outside of YOLO bounding boxes.
 *
 * This function iterates through a list of points and classifies each point as either inside or outside
 * any of the provided YOLO bounding boxes. Points that fall inside any of the YOLO bounding boxes are added
 * to the `max_corr_points_inside_yolo` vector, while points that fall outside are added to the `max_corr_points_outside_yolo` vector.
 *
 * @param[in] yolo_boxes A vector of `cv::Rect` objects representing the YOLO bounding boxes.
 * @param[in] max_corr_points A vector of `cv::Point` objects representing the points to be classified.
 * @param[out] max_corr_points_inside_yolo A vector to store points that fall inside any YOLO bounding box.
 * @param[out] max_corr_points_outside_yolo A vector to store points that fall outside all YOLO bounding boxes.
 */
void classifyPointsByYoloBoxes(const std::vector<cv::Rect>& yolo_boxes,const std::vector<cv::Point>& max_corr_points,
    std::vector<cv::Point>& max_corr_points_inside_yolo, std::vector<cv::Point>& max_corr_points_outside_yolo)
{
    max_corr_points_inside_yolo.clear();
    max_corr_points_outside_yolo.clear();

    auto is_point_in_boxes = [&yolo_boxes](const cv::Point& point)
    {
    	return std::any_of(yolo_boxes.begin(), yolo_boxes.end(), [&point](const cv::Rect& box) { return box.contains(point); });
    };

    for (const auto& point : max_corr_points)
        is_point_in_boxes(point) ? max_corr_points_inside_yolo.push_back(point) : max_corr_points_outside_yolo.push_back(point);
}


/**
 * @brief Groups points by the YOLO bounding box they fall into.
 *
 * This function iterates through a list of points and groups them by the YOLO bounding box they fall into.
 * Each YOLO bounding box is associated with a vector of points that are contained within it.
 *
 * @param[in] yolo_boxes A vector of `cv::Rect` objects representing the YOLO bounding boxes.
 * @param[in] points A vector of `cv::Point` objects representing the points to be grouped.
 * @return An unordered map where the key is the index of the YOLO bounding box and the value is a vector of `cv::Point` objects that fall within that bounding box.
 */
std::unordered_map<int, std::vector<cv::Point>> groupPointsByYoloBox(const std::vector<cv::Rect>& yolo_boxes, const std::vector<cv::Point>& points)
{
    std::unordered_map<int, std::vector<cv::Point>> points_in_boxes;

    for (size_t i = 0; i < yolo_boxes.size(); ++i)
        points_in_boxes.emplace(i, std::vector<cv::Point>());

    auto is_point_in_box = [](const cv::Point& point, const cv::Rect& box)
    {
    	return box.contains(point);
    };

    for (const auto& point : points)
    {
        for (size_t i = 0; i < yolo_boxes.size(); ++i)
        {
            if (is_point_in_box(point, yolo_boxes[i]))
            {
                points_in_boxes[i].push_back(point);
                break;
            }
        }
    }
    return points_in_boxes;
}


/**
 * @brief Associates YOLO bounding boxes with Regions of Interest (ROIs) generated from points.
 *
 * This function associates each YOLO bounding box with a set of ROIs generated from points that fall within the bounding box.
 * The ROIs are generated using predefined sizes calculated from clusters.
 *
 * @param[in] yolo_boxes A vector of `cv::Rect` objects representing the YOLO bounding boxes.
 * @param[in] max_corr_tp A vector of `cv::Point` objects representing the points to be associated with ROIs.
 * @return A vector of pairs, where each pair consists of a YOLO bounding box (cv::Rect) and a vector of associated ROIs (cv::Rect).
 *
 * @note The function groups points by their corresponding YOLO bounding box and generates ROIs for each group of points.
 *       The ROIs are generated using the sizes calculated from the clusters in the "kmeans_by_size" directory.
 *
 * @see groupPointsByYoloBox
 * @see listDirectories
 * @see calculateAvgDims
 * @see generateRoisFromPoints
 */
std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> associateYoloBoxesWithRois(const std::vector<cv::Rect>& yolo_boxes, const std::vector<cv::Point>& max_corr_tp)
{
    std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> yoloBox_roi_pairs;
    auto points_in_boxes = groupPointsByYoloBox(yolo_boxes, max_corr_tp);

    for (size_t i = 0; i < yolo_boxes.size(); ++i)
    {
        const auto& yolo_box = yolo_boxes[i];
        const auto& points = points_in_boxes[i];

        // Use the generateRoisFromPoints function to generate ROIs
        std::vector<std::string> kmeans_by_size_clusters;
        listDirectories(std::filesystem::path(SRC_DIR_PATH) / "kmeans_by_size", kmeans_by_size_clusters);

        std::vector<cv::Size> roi_sizes;
        roi_sizes.reserve(kmeans_by_size_clusters.size());
        for (size_t i = 0; i < kmeans_by_size_clusters.size(); i++)
            roi_sizes.push_back(calculateAvgDims(std::filesystem::path(kmeans_by_size_clusters[i])));


        std::vector<cv::Rect> rois = generateRoisFromPoints(points, roi_sizes);

        yoloBox_roi_pairs.emplace_back(yolo_box, rois);
    }
    return yoloBox_roi_pairs;
}

/**
 * @brief Selects the Regions of Interest (ROIs) with the highest Intersection over Union (IoU) for each YOLO bounding box.
 *
 * This function iterates over pairs of YOLO bounding boxes and associated ROIs, and selects the ROI with the highest IoU
 * for each YOLO bounding box. The selected ROIs are returned in a vector.
 *
 * @param[in] yoloBox_roi_pairs A vector of pairs, where each pair consists of a YOLO bounding box (cv::Rect) and a vector of associated ROIs (cv::Rect).
 * @return A vector of `cv::Rect` objects representing the ROIs with the highest IoU for each YOLO bounding box.
 *
 * @note The function calculates the IoU for each ROI and selects the ROI with the maximum IoU for each YOLO bounding box.
 *       If no valid ROI is found for a YOLO bounding box, that box is skipped.
 */
std::vector<cv::Rect> selectROIsWithHighestIoU(const std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>>& yoloBox_roi_pairs)
{
    std::vector<cv::Rect> rois_labels;

    for (const auto& pair : yoloBox_roi_pairs)
    {
        const cv::Rect& yolo_box = pair.first;
        const std::vector<cv::Rect>& rois = pair.second;

        double max_iou = 0.0;
        cv::Rect max_iou_roi;
        bool found_valid_roi = false;

        // Find the ROI with the maximum IoU
        for (const auto& roi : rois)
        {
            cv::Rect intersection = roi & yolo_box;
            double intersection_area = intersection.area();
            double union_area = roi.area() + yolo_box.area() - intersection_area;
            double iou = intersection_area / union_area;


            if (iou > max_iou)
            {
                max_iou = iou;
                max_iou_roi = roi;
                found_valid_roi = true;
            }
        }


        if (found_valid_roi)
        {
            rois_labels.emplace_back(max_iou_roi);
        }
    }

    return rois_labels;
}