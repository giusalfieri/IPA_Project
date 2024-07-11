#include "detection.h"

#include "utils.h"
#include <algorithm> // for std::max_element
#include "hog_features_extraction.h"


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



std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> associateYoloBoxesWithRois(const std::vector<cv::Rect>& yolo_boxes,const std::vector<cv::Point>& max_corr_tp)
{
    std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> yoloBox_roi_pairs;
    auto points_in_boxes = groupPointsByYoloBox(yolo_boxes, max_corr_tp);

    for (size_t i = 0; i < yolo_boxes.size(); ++i)
    {
        const auto& yolo_box = yolo_boxes[i];
        const auto& points = points_in_boxes[i];

        // Use the generateRoisFromPoints function to generate ROIs
        // N.B HERE roi_sizes IS A GLOBAL VARIABLE !!!!!!
        std::vector<cv::Rect> rois = generateRoisFromPoints(points, roi_sizes);

        yoloBox_roi_pairs.emplace_back(yolo_box, rois);
    }
    return yoloBox_roi_pairs;
}




std::vector<std::pair<cv::Rect, std::string>> labelRoisWithMaxIouAboveThreshold(const std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>>& yoloBox_roi_pairs,double iou_threshold)
{
    std::vector<std::pair<cv::Rect, std::string>> rois_labels;

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

        // Assign label only if a valid ROI was found
        if (found_valid_roi)
        {
            rois_labels.emplace_back(max_iou_roi, "TP");
        }
    }

    return rois_labels;
}



std::vector<std::pair<cv::Rect, std::string>> generateRoisWithLabelsFromPoints_Outside_Yolo(const std::vector<cv::Point>& max_corr_fp)
{
    const std::vector<cv::Rect> rois = generateRoisFromPoints(max_corr_fp, roi_sizes);

    // Preallocate capacity for rois_labels
    std::vector<std::pair<cv::Rect, std::string>> rois_labels;
    rois_labels.reserve(rois.size());

    for (const auto& roi : rois)
        rois_labels.emplace_back(roi, "FP");

    return rois_labels;
}


std::vector<std::pair<cv::Rect, std::string>> concatenateRoiLabelPairs(const std::vector<std::pair<cv::Rect, std::string>>& roi_labels_pairs_point_in_yolo,const std::vector<std::pair<cv::Rect, std::string>>& roi_labels_pairs_point_outside_yolo)
{
    std::vector<std::pair<cv::Rect, std::string>> concatenated_pairs;
    concatenated_pairs.reserve(roi_labels_pairs_point_in_yolo.size() + roi_labels_pairs_point_outside_yolo.size());
    concatenated_pairs.insert(concatenated_pairs.end(), roi_labels_pairs_point_in_yolo.begin(), roi_labels_pairs_point_in_yolo.end());
    concatenated_pairs.insert(concatenated_pairs.end(), roi_labels_pairs_point_outside_yolo.begin(), roi_labels_pairs_point_outside_yolo.end());

	return concatenated_pairs;
}


void saveRoiLabelPairsToFile(const std::vector<std::pair<cv::Rect, std::string>>& roi_labels, const std::string& filename)
{
    auto file = openFile(filename);

    try
    {
        for (const auto& [roi, label] : roi_labels)
            file << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << "," << label << "\n";

        file.flush();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Error writing to file: ") + e.what());
    }
}


 // Funzione per estrarre le ROI dalle coppie ROI-Label
std::vector<cv::Rect> extractRois(const std::vector<std::pair<cv::Rect, std::string>>& roi_labels_pairs)
{
    std::vector<cv::Rect> rois;
    rois.reserve(roi_labels_pairs.size());
    for (const auto& [roi, label] : roi_labels_pairs)
    {
        rois.push_back(roi);
    }
    return rois;
}







void readImageAndAnnotations(const std::string& testing_img_id, cv::Mat& src_img, cv::Mat& src_img_gray, std::vector<cv::Rect>& yolo_boxes)
{
    const std::string image_path = std::string(TESTING_DATASET_PATH) + "/" + testing_img_id + ".jpg";
    const std::string yolo_boxes_path = std::string(TESTING_DATASET_PATH) + "/" + testing_img_id + ".txt";

    src_img = cv::imread(image_path);
    if (!src_img.data)
        std::cerr << "Error: Could not read " << testing_img_id << ".jpg file\n";

    // Template matching will be performed on the grayscale version of the image
    // The templates are also in grayscale
    cv::cvtColor(src_img, src_img_gray, cv::COLOR_BGR2GRAY);

    try
    {
        yolo_boxes = readYoloBoxes(std::filesystem::path(yolo_boxes_path));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
    }
}



std::vector<std::pair<cv::Rect, std::string>> classifyAndLabelRois(const std::vector<cv::Rect>& yolo_boxes, const std::vector<cv::Point>& max_corr_points)
{
    // Divide the points into two groups: the ones inside YOLO boxes and the ones outside any YOLO box
    std::vector<cv::Point> max_corr_points_inside_yolo;
    std::vector<cv::Point> max_corr_points_outside_yolo;
    classifyPointsByYoloBoxes(yolo_boxes, max_corr_points, max_corr_points_inside_yolo, max_corr_points_outside_yolo);

    // Associate each YOLO box with the ROIs extracted from the points inside it
    // The result is a vector of pairs, where each pair contains a YOLO box and the ROIs associated with it
    std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> yoloBox_roi_pairs = associateYoloBoxesWithRois(yolo_boxes, max_corr_points_inside_yolo);

    // For each YOLO box, label as "TP" the ROI having the maximum IOU (above a threshold) with the YOLO box
    // All others ROis for that YOLO box are discarded
    constexpr double iou_threshold = 0.1;
    std::vector<std::pair<cv::Rect, std::string>> rois_labels_for_points_inside_yolo = labelRoisWithMaxIouAboveThreshold(yoloBox_roi_pairs, iou_threshold);

    // Generate ROIs around points outside YOLO boxes and label them as "FP"
    std::vector<std::pair<cv::Rect, std::string>> rois_labels_for_points_outside_yolo = generateRoisWithLabelsFromPoints_Outside_Yolo(max_corr_points_outside_yolo);

    // Concatenate ROI-label pairs
    return concatenateRoiLabelPairs(rois_labels_for_points_inside_yolo, rois_labels_for_points_outside_yolo);
}



void saveResults(const std::string& testing_img_id, const std::vector<std::pair<cv::Rect, std::string>>& rois_label_pairs_all_points, const cv::Mat& src_img_gray,
    const std::filesystem::path& output_dir)
{
    // Create output directory for saving results
    //std::filesystem::path output_dir = createDirectory(std::filesystem::path(SRC_DIR_PATH), "classification");

    // Save ROI-label pairs to file
    saveRoiLabelPairsToFile(rois_label_pairs_all_points, (output_dir / "roi_label_pairs.csv").string());

    // From the ROI-label pairs, extract only the ROIs
    std::vector<cv::Rect> rois = extractRois(rois_label_pairs_all_points);

    // HOG features extraction from the previously extracted ROIs
    std::vector<std::vector<float>> vector_of_hog_features = hog_features_extraction(rois, src_img_gray);

    // Write HOG features to CSV file
    std::filesystem::path testing_samples_path = output_dir / "testing_samples.csv";
    writeHogFeaturesToCsv(vector_of_hog_features, testing_samples_path.string());

    // Write the testing_img_id to a file
    try
    {
        std::filesystem::path img_id_path = output_dir / "testing_img_id.txt";
        std::ofstream img_id_file(img_id_path);
        if (!img_id_file)
        {
            throw std::ios_base::failure("Error: Could not open file to write testing_img_id");
        }
        img_id_file << testing_img_id;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
    }
}

void drawAndSaveResults(const cv::Mat& src_img, const std::vector<std::pair<cv::Rect, std::string>>& rois_label_pairs_all_points, const std::filesystem::path& output_dir)
{
    // Draw the detection results on a copy of the source color image
    cv::Mat src_img_drawing = src_img.clone();
    drawRectangles(src_img_drawing, rois_label_pairs_all_points);
    imshow("Detection result", src_img_drawing, true, 0.4f);
    cv::imwrite((output_dir / "result.png").string(), src_img_drawing);
}