#include "template_extraction.h"

#include <unordered_set>

#include "utils.h"


void selectAirplanes(const cv::Mat& img, const std::vector<cv::Rect>& yolo_boxes, std::vector<cv::Rect>& selected_airplanes_yolo_boxes, int& count, const std::string& img_filename, const std::filesystem::path& extracted_templates_folder);

void binarizeAirplanes(const cv::Mat& channel, const std::vector<cv::Rect>& selected_airplanes_yolo_boxes, std::vector<cv::Mat>& bin_airplanes);

void calculateGeometricMoments(const std::vector<cv::Mat>& bin_airplanes, const std::vector<cv::Rect>& yolo_boxes, std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors);

void saveTemplates(const std::vector<cv::Mat>& templates_vector, int& count, const std::string& img_filename, const std::filesystem::path& extracted_templates_folder);

void saveTemplate(const cv::Mat& airplane, int& count, const std::string& img_filename, const std::filesystem::path& extracted_templates_folder);

void extractTemplatesFromAirplanes(const cv::Mat& img, const std::vector<cv::Mat>& bin_airplanes, const std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors, const std::vector<cv::Rect>& yolo_boxes, std::vector<cv::Mat>& templates_vector);

std::string getValidInput(const std::string& prompt, const std::vector<std::string>& valid_inputs);

cv::Mat getBoundarySafeROI(const cv::Mat& img, cv::Rect& roi);




void extractTemplates()
{
    std::vector<std::string> dataset_img_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.jpg", dataset_img_paths);

    std::vector<std::string> yolo_labels_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.txt", yolo_labels_paths);

    auto extracted_templates_folder = createDirectory(std::filesystem::path(TRAINING_DATASET_PATH).parent_path(), "extracted_templates");

    int count = 0;

    for (size_t k = 0; k < dataset_img_paths.size(); ++k) 
    {
        std::cout << "\n---------------------------------------\n";
        std::cout << "      Starting processing image " << k;
        std::cout << "\n---------------------------------------\n\n";

        cv::Mat img = cv::imread(dataset_img_paths[k]);
        std::filesystem::path p(dataset_img_paths[k]);
        auto img_filename = p.stem().string();

        cv::Mat img_HSV;
        cv::cvtColor(img, img_HSV, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> channels;
        cv::split(img_HSV, channels);

        std::vector<cv::Rect> yolo_boxes;
        processYoloLabels(yolo_labels_paths[k], img, yolo_boxes);

        std::vector<cv::Rect> selected_airplanes_yolo_boxes;
        selectAirplanes(img, yolo_boxes, selected_airplanes_yolo_boxes, count, img_filename, extracted_templates_folder);

        std::vector<cv::Mat> bin_airplanes;
        binarizeAirplanes(channels[2], selected_airplanes_yolo_boxes, bin_airplanes);

        std::vector<std::pair<cv::Point2f, double>> geometric_moments_descriptors;
        calculateGeometricMoments(bin_airplanes, yolo_boxes, geometric_moments_descriptors);

        std::vector<cv::Mat> templates_vector;
        extractTemplatesFromAirplanes(img, bin_airplanes, geometric_moments_descriptors, yolo_boxes, templates_vector);

        saveTemplates(templates_vector, count, img_filename, extracted_templates_folder);

        std::cout << "\n---------------------------------------\n";
        std::cout << "Image " << k << " has been processed successfully";
        std::cout << "\n---------------------------------------\n\n";
    }
}

void selectAirplanes(const cv::Mat& img, const std::vector<cv::Rect>& yolo_boxes, std::vector<cv::Rect>& selected_airplanes_yolo_boxes, int& count, const std::string& img_filename, const std::filesystem::path& extracted_templates_folder)
{
    for (const auto& box : yolo_boxes)
    {
        cv::Mat airplane = img(box).clone();
        cv::imshow("Airplane", airplane);
        cv::waitKey(100);
        std::string answer = getValidInput("Process this airplane?(Y/y,N/n): ", { "Y", "y", "N", "n" });

        if (answer == "Y" || answer == "y") 
        {
            selected_airplanes_yolo_boxes.push_back(box);
        }
        else
        {
            std::string new_answer = getValidInput("Perform rotation by a multiple of 90°(CW)?", { "Y", "y", "N", "n" });

            if (new_answer == "Y" || new_answer == "y")
            {
                new_answer = getValidInput("Please provide a rotation type (0-->no rotation, 1-->CW rotation 90, 2-->CW rotation 180, 3-->CW rotation 270): ", { "0", "1", "2", "3" });
                airplane = rotate90(airplane, std::stoi(new_answer));
            }
            saveTemplate(airplane, count, img_filename, extracted_templates_folder);
        }
    }
}

void binarizeAirplanes(const cv::Mat& channel, const std::vector<cv::Rect>& selected_airplanes_yolo_boxes, std::vector<cv::Mat>& bin_airplanes)
{
    for (const auto& box : selected_airplanes_yolo_boxes) 
    {
        cv::Mat airplane = channel(box).clone();
        cv::Mat img_bin_adaptive;
        int block_size = airplane.cols;
        if (block_size % 2 == 0)
            block_size += 1;
        constexpr int C = -20;
        cv::adaptiveThreshold(airplane, img_bin_adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, C);
        cv::morphologyEx(img_bin_adaptive, img_bin_adaptive, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        bin_airplanes.push_back(img_bin_adaptive);
    }
}

void calculateGeometricMoments(const std::vector<cv::Mat>& bin_airplanes, const std::vector<cv::Rect>& yolo_boxes, std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors)
{
    for (size_t i = 0; i < bin_airplanes.size(); ++i) 
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin_airplanes[i], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        std::sort(contours.begin(), contours.end(), sortByDescendingArea);

        cv::Moments moments = cv::moments(contours[0], true);
        cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);
        double angle = 0.5 * std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

        cv::Point2f corrected_center = cv::Point2f(center.x + yolo_boxes[i].x, center.y + yolo_boxes[i].y);
        geometric_moments_descriptors.push_back(std::make_pair(corrected_center, rad2degrees(angle)));
    }
}

// Funzione helper per il calcolo degli angoli corretti
double correctAngle(double angle)
{
    if (angle >= 0 && angle <= 90)
        return angle + 90.0f;
    else  if (angle < 0 && angle >= -90)
        return 90.0f - (-angle);
}

// Funzione helper per il calcolo del ROI centrato
cv::Rect calculateCenteredROI(const cv::Point2f& center, const cv::Size& size, float scale_factor)
{
    return cv::Rect
	(
        static_cast<int>(std::round(center.x - (size.width * scale_factor) / 2.0f)),
        static_cast<int>(std::round(center.y - (size.height * scale_factor) / 2.0f)),
        static_cast<int>(std::round(size.width * scale_factor)),
        static_cast<int>(std::round(size.height * scale_factor))
    );
}

void extractTemplatesFromAirplanes(const cv::Mat& img, const std::vector<cv::Mat>& bin_airplanes,
    const std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors,
    const std::vector<cv::Rect>& yolo_boxes, std::vector<cv::Mat>& templates_vector)
{
    constexpr float finalTemplateScalingFactor = 1.15;
    constexpr float roiExpansionFactor = 2.0f;

    for (size_t i = 0; i < bin_airplanes.size(); ++i)
    {
        // Validazione degli indici per evitare accessi fuori dai limiti
        if (i >= geometric_moments_descriptors.size() || i >= yolo_boxes.size())
        {
            std::cerr << "Invalid index: " << i << std::endl;
            continue; // Passa al prossimo elemento
        }

        cv::Point2f center = geometric_moments_descriptors[i].first;
        double angle = correctAngle(geometric_moments_descriptors[i].second);

        cv::Rect roi = calculateCenteredROI(center, yolo_boxes[i].size(), roiExpansionFactor);
        cv::Mat rotation_roi = getBoundarySafeROI(img, roi);

        cv::Point roi_center(rotation_roi.cols / 2, rotation_roi.rows / 2);
        cv::Mat rotation_mat = cv::getRotationMatrix2D(roi_center, angle, 1);

        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), rotation_roi.size(), angle).boundingRect2f();
        rotation_mat.at<double>(0, 2) += bbox.width / 2.0f - rotation_roi.cols / 2.0f;
        rotation_mat.at<double>(1, 2) += bbox.height / 2.0f - rotation_roi.rows / 2.0f;

        cv::Mat dst;
        cv::warpAffine(rotation_roi, dst, rotation_mat, bbox.size());

        cv::Rect final_roi(
            static_cast<int>(std::round(dst.cols / 2.0f - (yolo_boxes[i].width * finalTemplateScalingFactor) / 2.0f)),
            static_cast<int>(std::round(dst.rows / 2.0f - (yolo_boxes[i].height * finalTemplateScalingFactor) / 2.0f)),
            static_cast<int>(std::round(yolo_boxes[i].width * finalTemplateScalingFactor)),
            static_cast<int>(std::round(yolo_boxes[i].height * finalTemplateScalingFactor))
        );

        // Verifica dei limiti di final_roi prima di usarlo
        if (final_roi.x >= 0 && final_roi.y >= 0 && final_roi.x + final_roi.width <= dst.cols && final_roi.y + final_roi.height <= dst.rows)
        {
            cv::Mat final_template = dst(final_roi);
            templates_vector.push_back(final_template);
        }
        else
            std::cerr << "Invalid ROI for final template: " << final_roi << std::endl;

    }
}

void saveTemplates(const std::vector<cv::Mat>& templates_vector, int& count, const std::string& img_filename, const std::filesystem::path& extracted_templates_folder)
{
    for (const auto& airplane_template : templates_vector) 
    {
        cv::imshow("Template", airplane_template);
        cv::waitKey(100);
        cv::Mat airplane = airplane_template.clone();

        auto answer = getValidInput("Do you wish to keep this template?(Y/y,N/n): ", { "Y", "y", "N", "n" });
        if (answer == "Y" || answer == "y") 
        {
            answer = getValidInput("Perform rotation?(Y/y,N/n)\n", { "Y", "y", "N", "n" });
            if (answer == "Y" || answer == "y") 
            {
                answer = getValidInput("Please provide a rotation type (0-->no rotation, 1-->CW rotation 90, 2-->CW rotation 180, 3-->CW rotation 270): ", { "0", "1", "2", "3" });
                airplane = rotate90(airplane_template, std::stoi(answer));
            }
        }
        else continue;

        const auto template_name = "template_" + std::to_string(count++) + "_" + img_filename;
        std::filesystem::path output_path = extracted_templates_folder / (template_name + ".png");
        cv::imwrite(output_path.string(), airplane);
    }
}


void saveTemplate(const cv::Mat& airplane, int& count, const std::string& img_filename, const std::filesystem::path& extracted_templates_folder)
{
    const auto template_name = "template_" + std::to_string(count++) + "_" + img_filename;
    const std::filesystem::path output_path = extracted_templates_folder / (template_name + ".png");
    cv::imwrite(output_path.string(), airplane);
}


std::string getValidInput(const std::string& prompt, const std::vector<std::string>& valid_inputs)
{
    std::unordered_set<std::string> valid_set(valid_inputs.begin(), valid_inputs.end());
    std::string answer;

    do
    {
        std::cout << prompt;
        std::cin >> answer;

        if (valid_set.find(answer) == valid_set.end()) 
        {
            std::cout << "Invalid input! Please enter one of the following: ";
            for (const auto& input : valid_inputs)
                std::cout << input << " ";

            std::cout << "\n";
        }

    } while (valid_set.find(answer) == valid_set.end());

    return answer;
}


cv::Mat getBoundarySafeROI(const cv::Mat& img, cv::Rect& roi)
{
    // Check if the ROI is within the image boundaries
    if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= img.cols && roi.y + roi.height <= img.rows) 
        return img(roi);

    // If the ROI is out of the image boundaries, pad the image
    const int top = std::max(-roi.y, 0);
    const int bottom = std::max(roi.y + roi.height - img.rows, 0);
    const int left = std::max(-roi.x, 0);
    const int right = std::max(roi.x + roi.width - img.cols, 0);

    cv::Mat padding_clone;
    cv::copyMakeBorder(img, padding_clone, top, bottom, left, right, cv::BORDER_REFLECT, 0);

    // Adjust ROI to the padded image
    roi.x += left;
    roi.y += top;

    // Apply the ROI to the padded image
    return padding_clone(roi);
}