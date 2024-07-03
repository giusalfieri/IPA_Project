#include "svm_inference_input.h"
#include "utils.h"
#include <algorithm> // for std::max_element



void classifyPointsByYoloBoxes(const std::vector<cv::Rect>& yolo_boxes,
                               const std::vector<cv::Point>& max_corr_points,
                               std::vector<cv::Point>& max_corr_tp,
                               std::vector<cv::Point>& max_corr_fp)
{
    max_corr_tp.clear();
    max_corr_fp.clear();

    auto is_point_in_boxes = [&yolo_boxes](const cv::Point& point)
    {
    	return std::any_of(yolo_boxes.begin(), yolo_boxes.end(), [&point](const cv::Rect& box) { return box.contains(point); });
    };

    for (const auto& point : max_corr_points)
        is_point_in_boxes(point) ? max_corr_tp.push_back(point) : max_corr_fp.push_back(point);
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
        std::vector<cv::Rect> rois = generateRoisFromPoints(points, roi_sizes);

        yoloBox_roi_pairs.emplace_back(yolo_box, rois);
    }
    return yoloBox_roi_pairs;
}



std::vector<std::pair<cv::Rect, std::string>> labelRoisBasedOnIouWithYoloBoxes(const std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>>& yoloBox_roi_pairs, double iou_threshold)
{
    std::vector<std::pair<cv::Rect, std::string>> rois_labels;

    //for (const auto& [yolo_box, rois] : yoloBox_roi_pairs)
    for (const auto& pair : yoloBox_roi_pairs)
    {
        const cv::Rect& yolo_box = pair.first;
        const std::vector<cv::Rect>& rois = pair.second;
        // Find the ROI with the maximum IoU above the threshold
        auto max_iou_it = std::max_element(rois.begin(), rois.end(),
            [&](const cv::Rect& a, const cv::Rect& b) {
                const cv::Rect intersection_a = a & yolo_box;
                const cv::Rect union_a = a | yolo_box;
                const double iou_a = static_cast<double>(intersection_a.area()) / union_a.area();

                const cv::Rect intersection_b = b & yolo_box;
                const cv::Rect union_b = b | yolo_box;
                const double iou_b = static_cast<double>(intersection_b.area()) / union_b.area();

                return iou_a < iou_b;
            });

        // Assign labels based on the IoU values
        for (const auto& roi : rois)
        {
            cv::Rect intersection = roi & yolo_box;
            cv::Rect union_rect = roi | yolo_box;
            double intersection_area = intersection.area();
            double union_area = union_rect.area();
            double iou = static_cast<double>(intersection_area) / union_area;

            if (max_iou_it != rois.end() && roi == *max_iou_it && iou >= iou_threshold)
                rois_labels.emplace_back(roi, "TP");
            else
                rois_labels.emplace_back(roi, "FP");
            
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