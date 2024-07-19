#include "svm_training.h"


#include "utils.h"
#include "hog_features_extraction.h"
#include <random>
#include <filesystem>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "template_matching.h"



void classifyPointsByYoloBoxes(const std::vector<cv::Rect>& yolo_boxes,
    const std::vector<cv::Point>& max_corr_points,
    std::vector<cv::Point>& max_corr_points_inside_yolo,
    std::vector<cv::Point>& max_corr_points_outside_yolo);
std::vector<cv::Rect> myfunc(const std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>>& yoloBox_roi_pairs, double iou_threshold);
std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> associateYoloBoxesWithRois(const std::vector<cv::Rect>& yolo_boxes, const std::vector<cv::Point>& max_corr_tp);



// ============================================================================= 
// =============================================================================
//                               SVM TRAINING 
// =============================================================================
// =============================================================================

// Extracts HOG features from images in the training dataset and saves them to CSV files for SVM training
void extract_csv_for_svm_cross_validation()
{
   
    std::vector<std::string> kmeans_by_size_clusters;
    listDirectories(std::filesystem::path(SRC_DIR_PATH) / "kmeans_by_size", kmeans_by_size_clusters);


    std::vector<cv::Size> roi_sizes;
    roi_sizes.reserve(kmeans_by_size_clusters.size());
    for (size_t i = 0; i < kmeans_by_size_clusters.size(); i++)
        roi_sizes.push_back(calculateAvgDims(std::filesystem::path(kmeans_by_size_clusters[i])));


    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, roi_sizes.size() - 1); // Uniform distribution between 0 and roi_sizes.size() - 1


    
    std::vector<std::string> dataset_img_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.jpg", dataset_img_paths);
    // Read yolo labels paths for the dataset images
    std::vector<std::string> yolo_labels_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.txt", yolo_labels_paths);


    // Read images all training images in grayscale
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
        std::vector<cv::Rect> yolo_boxes = readYoloBoxes(yolo_labels_paths[i] );


        // Classify points by their position inside or outside YOLO boxes
        std::vector<cv::Point> max_corr_points_inside_yolo;
    	std::vector<cv::Point> max_corr_points_outside_yolo;
        classifyPointsByYoloBoxes(yolo_boxes, matched_points, max_corr_points_inside_yolo, max_corr_points_outside_yolo);


        // Filter outside yolo points by minimum distance between them
        std::vector<cv::Point> max_corr_points_out_yolo = filterPointsByMinDistance(max_corr_points_outside_yolo, 100);


    	// Associate each YOLO box with the ROIs extracted from the points inside it
        // The result is a vector of pairs, where each pair contains a YOLO box and the ROIs associated with it
        std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> yoloBox_roi_pairs = associateYoloBoxesWithRois(yolo_boxes, max_corr_points_inside_yolo);


        // For each YOLO box, label as "TP" the ROI having the maximum IOU (above a threshold) with the YOLO box
        // All others ROis for that YOLO box are discarded
        constexpr double iou_threshold = 0.0;
        //std::vector<std::pair<cv::Rect, std::string>> rois_labels_for_points_inside_yolo = labelRoisWithMaxIouAboveThreshold(yoloBox_roi_pairs, iou_threshold);
        std::vector<cv::Rect> tp_rois = myfunc(yoloBox_roi_pairs, iou_threshold);
        
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


        // ONLY FOR DEBUGGING PURPOSES
        /*
        // Draw the ROIs on the current image
        cv::Mat img_with_rois = src_imgs_gray[i].clone();
        cv::cvtColor(src_imgs_gray[i], img_with_rois, cv::COLOR_GRAY2BGR);

        // Draw the true positive ROIs in green
        for (const auto& roi : tp_rois)
            cv::rectangle(img_with_rois, roi, cv::Scalar(0, 255, 0), 2);
        

        // Draw the false positive ROIs in red
        for (const auto& roi : fp_rois)
            cv::rectangle(img_with_rois, roi, cv::Scalar(0, 0, 255), 2);

    
        imshow("ROIs", img_with_rois, true, 0.2f);
        cv::waitKey(0);
        */

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



void classifyPointsByYoloBoxes(const std::vector<cv::Rect>& yolo_boxes,
    const std::vector<cv::Point>& max_corr_points,
    std::vector<cv::Point>& max_corr_points_inside_yolo,
    std::vector<cv::Point>& max_corr_points_outside_yolo)
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
        std::vector<cv::Size> roi_sizes(kmeans_by_size_clusters.size());
        for (size_t i = 0; i < kmeans_by_size_clusters.size(); i++)
            roi_sizes.emplace_back(calculateAvgDims(std::filesystem::path(kmeans_by_size_clusters[i])));



        std::vector<cv::Rect> rois = generateRoisFromPoints(points, roi_sizes);

        yoloBox_roi_pairs.emplace_back(yolo_box, rois);
    }
    return yoloBox_roi_pairs;
}


std::vector<cv::Rect> myfunc(const std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>>& yoloBox_roi_pairs, double iou_threshold)
{
    std::vector<cv::Rect> rois_labels;

    for (const auto& pair : yoloBox_roi_pairs)
    {
        const cv::Rect& yolo_box = pair.first;
        const std::vector<cv::Rect>& rois = pair.second;

        double max_iou = 0.0;
        cv::Rect max_iou_roi;
        bool found_valid_roi = false;

        // Find the ROI with the maximum IoU above the threshold
        for (const auto& roi : rois)
        {
            cv::Rect intersection = roi & yolo_box;
            double intersection_area = intersection.area();
            double union_area = roi.area() + yolo_box.area() - intersection_area;
            double iou = intersection_area / union_area;


            if (iou >= iou_threshold && iou > max_iou)
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