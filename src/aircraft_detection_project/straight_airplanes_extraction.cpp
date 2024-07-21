#include "straight_airplanes_extraction.h"
#include "utils.h"
#include <unordered_set>



//======================================================================================================
//                                      Forward Declarations
void selectAirplanes(const cv::Mat& img, const std::vector<cv::Rect>& yolo_boxes, std::vector<cv::Rect>& selected_airplanes_yolo_boxes, int& count, const std::string& img_filename, const std::filesystem::path& straight_airplanes_folder);

void binarizeAirplanes(const cv::Mat& channel, const std::vector<cv::Rect>& selected_airplanes_yolo_boxes, std::vector<cv::Mat>& bin_airplanes);

void calculateGeometricMoments(const std::vector<cv::Mat>& bin_airplanes, const std::vector<cv::Rect>& yolo_boxes, std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors);

void saveStraightAirplanes(const std::vector<cv::Mat>& airplanes, int& count, const std::string& img_filename, const std::filesystem::path& straight_airplanes_folder);

void saveAirplane(const cv::Mat& airplane, int& count, const std::string& img_filename, const std::filesystem::path& straight_airplane_folder);

void extractRotatedAirplanes(const cv::Mat& img, const std::vector<cv::Mat>& bin_airplanes, const std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors, const std::vector<cv::Rect>& yolo_boxes, std::vector<cv::Mat>& airplanes_vector);

std::string getValidInput(const std::string& prompt, const std::vector<std::string>& valid_inputs);

cv::Mat getBoundarySafeROI(const cv::Mat& img, cv::Rect& roi);
//=======================================================================================================



void extractStraightAirplanes()
{
    std::vector<std::string> dataset_img_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.jpg", dataset_img_paths);

    std::vector<std::string> yolo_labels_paths;
    globFiles(TRAINING_DATASET_PATH, "/*.txt", yolo_labels_paths);

    auto straight_airplanes_folder = createDirectory(std::filesystem::path(SRC_DIR_PATH), "straight_airplanes");

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
        selectAirplanes(img, yolo_boxes, selected_airplanes_yolo_boxes, count, img_filename, straight_airplanes_folder);

        std::vector<cv::Mat> bin_airplanes;
        binarizeAirplanes(channels[2], selected_airplanes_yolo_boxes, bin_airplanes);

        std::vector<std::pair<cv::Point2f, double>> geometric_moments_descriptors;
        calculateGeometricMoments(bin_airplanes, yolo_boxes, geometric_moments_descriptors);

        std::vector<cv::Mat> straight_airplanes;
        extractRotatedAirplanes(img, bin_airplanes, geometric_moments_descriptors, yolo_boxes, straight_airplanes);

        saveStraightAirplanes(straight_airplanes, count, img_filename, straight_airplanes_folder);

        std::cout << "\n---------------------------------------\n";
        std::cout << "Image " << k << " has been processed successfully";
        std::cout << "\n---------------------------------------\n\n";
    }
}






void selectAirplanes(const cv::Mat& img, const std::vector<cv::Rect>& yolo_boxes, std::vector<cv::Rect>& selected_airplanes_yolo_boxes, int& count, const std::string& img_filename, const std::filesystem::path& straight_airplanes_folder)
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
            saveAirplane(airplane, count, img_filename, straight_airplanes_folder);
        }
    }
}

/**
 * @brief Binarizes the regions of interest (ROIs) of airplanes in an image channel.
 *
 * This function processes a specific channel of an image to binarize the regions corresponding
 * to the provided YOLO bounding boxes. Each ROI is extracted, binarized using adaptive thresholding,
 * and then morphologically dilated to enhance the binary image. The resulting binary images are
 * stored in the provided vector.
 *
 * @param channel                        The input image channel (grayscale) from which the airplane
 *                                       ROIs are extracted and binarized.
 * @param selected_airplanes_yolo_boxes  A vector of rectangles defining the bounding boxes of the
 *                                       airplanes to be binarized.
 * @param bin_airplanes                  A vector where the binarized airplane images will be stored.
 */
void binarizeAirplanes(const cv::Mat& channel, const std::vector<cv::Rect>& selected_airplanes_yolo_boxes, std::vector<cv::Mat>& bin_airplanes)
{
    for (const auto& box : selected_airplanes_yolo_boxes) 
    {
        cv::Mat airplane = channel(box).clone();
        cv::Mat img_bin_adaptive;

        int block_size = airplane.cols;
        if (block_size % 2 == 0)
        {
            block_size += 1;
        }

        constexpr int C = -20;
        cv::adaptiveThreshold(airplane, img_bin_adaptive, 255, 
            cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, C);

    	cv::morphologyEx(img_bin_adaptive, img_bin_adaptive, cv::MORPH_DILATE, 
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    	bin_airplanes.push_back(img_bin_adaptive);
    }
}

/**
 * @brief Calculates the geometric moments and the orientation angle for each binary airplane image.
 *
 * This function processes a vector of binary airplane images to calculate their geometric moments
 * and the orientation angle. For each binary image, it finds the contours, sorts them by area to
 * select the largest contour, and then computes the moments. The centroid of the largest contour
 * and the orientation angle are calculated and corrected by the corresponding YOLO bounding box offset.
 * The results are stored in a vector of pairs, each containing the centroid (as cv::Point2f) and
 * the orientation angle (in degrees).
 *
 * @param bin_airplanes                  A vector of binary images of airplanes.
 * @param yolo_boxes                     A vector of rectangles defining the bounding boxes of the airplanes.
 * @param geometric_moments_descriptors  A vector where the calculated centroids and orientation angles will be stored.
 */
void calculateGeometricMoments(
    const std::vector<cv::Mat>& bin_airplanes, 
    const std::vector<cv::Rect>& yolo_boxes, 
    std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors)
{
    for (size_t i = 0; i < bin_airplanes.size(); ++i) 
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin_airplanes[i], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        // Sort the contours by area in descending order
        std::sort(contours.begin(), contours.end(), sortByDescendingArea);

        cv::Moments moments = cv::moments(contours[0], true);
        cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);
        double angle = 0.5 * std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

        auto corrected_center = cv::Point2f(center.x + yolo_boxes[i].x, center.y + yolo_boxes[i].y);
        geometric_moments_descriptors.emplace_back(corrected_center, rad2degrees(angle));
    }
}

double correctAngle(double angle)
{
    if (angle >= 0 && angle <= 90)
        return angle + 90.0f;
    else 
        return 90.0f - (-angle);
}

/**
 * @brief Calculates a centered region of interest (ROI) around a given point with a specified scale factor.
 *
 * This function takes a center point, a size, and a scale factor to calculate a rectangular ROI
 * that is centered around the given point. The size of the ROI is scaled by the specified factor.
 *
 * @param center       The center point around which the ROI is calculated.
 * @param size         The original size of the ROI.
 * @param scale_factor The factor by which the size of the ROI is scaled.
 * @return cv::Rect    The calculated centered ROI.
 */
cv::Rect calculateCenteredROI(const cv::Point2f& center, const cv::Size& size, float scale_factor)
{
    return cv::Rect(
        static_cast<int>(std::round(center.x - (size.width * scale_factor) / 2.0f)),
        static_cast<int>(std::round(center.y - (size.height * scale_factor) / 2.0f)),
        static_cast<int>(std::round(size.width * scale_factor)),
        static_cast<int>(std::round(size.height * scale_factor))
    );
}

/**
 * @brief Extracts and rotates airplanes from an image based on their geometric moments and YOLO bounding boxes.
 *
 * This function processes an image to extract regions of interest (ROIs) corresponding to detected airplanes,
 * rotates these ROIs to align the airplanes based on their geometric moments, and stores the resulting images
 * in the provided vector.
 *
 * @param img                        The input image from which airplanes are extracted.
 * @param bin_airplanes              A vector of binary images of airplanes.
 * @param geometric_moments_descriptors A vector of pairs containing the centroid and orientation angle of each airplane.
 * @param yolo_boxes                 A vector of rectangles defining the bounding boxes of the airplanes.
 * @param airplanes_vector           A vector where the extracted and rotated airplane images will be stored.
 */
void extractRotatedAirplanes(const cv::Mat& img,
    const std::vector<cv::Mat>& bin_airplanes,
    const std::vector<std::pair<cv::Point2f, double>>& geometric_moments_descriptors,
    const std::vector<cv::Rect>& yolo_boxes,
    std::vector<cv::Mat>& airplanes_vector)
{
    constexpr float final_roi_scaling_factor = 1.15f;
    constexpr float roi_expansion_factor = 2.0f;

    for (size_t i = 0; i < bin_airplanes.size(); ++i)
    {
        // Check if the index is valid. If not, skip this airplane
        if (i >= geometric_moments_descriptors.size() || i >= yolo_boxes.size())
        {
            std::cerr << "Invalid index: " << i << "\n";
            continue;
        }

        cv::Point2f center = geometric_moments_descriptors[i].first;
        double angle = correctAngle(geometric_moments_descriptors[i].second);

        cv::Rect roi = calculateCenteredROI(center, yolo_boxes[i].size(), roi_expansion_factor);
        cv::Mat rotation_roi = getBoundarySafeROI(img, roi);

        cv::Point roi_center(rotation_roi.cols / 2, rotation_roi.rows / 2);
        cv::Mat rotation_mat = cv::getRotationMatrix2D(roi_center, angle, 1);

        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), rotation_roi.size(), angle).boundingRect2f();
        rotation_mat.at<double>(0, 2) += bbox.width / 2.0f - rotation_roi.cols / 2.0f;
        rotation_mat.at<double>(1, 2) += bbox.height / 2.0f - rotation_roi.rows / 2.0f;

        cv::Mat dst;
        cv::warpAffine(rotation_roi, dst, rotation_mat, bbox.size());

        cv::Rect final_roi(
            static_cast<int>(std::round(dst.cols / 2.0f - (yolo_boxes[i].width * final_roi_scaling_factor) / 2.0f)),
            static_cast<int>(std::round(dst.rows / 2.0f - (yolo_boxes[i].height * final_roi_scaling_factor) / 2.0f)),
            static_cast<int>(std::round(yolo_boxes[i].width * final_roi_scaling_factor)),
            static_cast<int>(std::round(yolo_boxes[i].height * final_roi_scaling_factor))
        );

        // Check if the final ROI is within the boundaries of the image
        // If not, skip this airplane
        if (final_roi.x >= 0 && final_roi.y >= 0 && final_roi.x + final_roi.width <= dst.cols && final_roi.y + final_roi.height <= dst.rows)
        {
            cv::Mat straight_airplane = dst(final_roi);
            airplanes_vector.push_back(straight_airplane);
        }
        else
        {
            std::cerr << "Invalid ROI for final airplane: " << final_roi << "\n";
        }
    }
}

void saveStraightAirplanes(const std::vector<cv::Mat>& airplanes, int& count, const std::string& img_filename, const std::filesystem::path& straight_airplanes_folder)
{
    for (const auto& airplane : airplanes) 
    {
        cv::imshow("Airplane", airplane);
        cv::waitKey(100);
        cv::Mat straight_airplane = airplane.clone();

        auto answer = getValidInput("Do you wish to keep this airplane?(Y/y,N/n): ", { "Y", "y", "N", "n" });
        if (answer == "Y" || answer == "y") 
        {
            answer = getValidInput("Perform rotation?(Y/y,N/n)\n", { "Y", "y", "N", "n" });
            if (answer == "Y" || answer == "y") 
            {
                answer = getValidInput("Please provide a rotation type (0-->no rotation, 1-->CW rotation 90, 2-->CW rotation 180, 3-->CW rotation 270): ", { "0", "1", "2", "3" });
                straight_airplane = rotate90(airplane, std::stoi(answer));
            }
        }
        else continue;

        const auto straight_airplane_name = "airplane_" + std::to_string(count++) + "_" + img_filename;
        std::filesystem::path output_path = straight_airplanes_folder / (straight_airplane_name + ".png");
        cv::imwrite(output_path.string(), straight_airplane);
    }
}


/**
 * @brief Saves a given airplane image to a specified folder with a unique filename.
 *
 * This function saves the provided airplane image to the specified folder. The filename
 * is generated using a counter, the original image filename, and a predefined format.
 *
 * @param airplane                  The airplane image to be saved.
 * @param count                     A reference to the counter used for generating unique filenames.
 * @param img_filename              The original filename of the image.
 * @param straight_airplane_folder  The folder where the airplane image will be saved.
 */
void saveAirplane(const cv::Mat& airplane,
    int& count,
    const std::string& img_filename,
    const std::filesystem::path& straight_airplane_folder)
{
    const auto airplane_name = "airplane_" + std::to_string(count++) + "_" + img_filename;
    const std::filesystem::path output_path = straight_airplane_folder / (airplane_name + ".png");
    cv::imwrite(output_path.string(), airplane);
}


/**
 * @brief Prompts the user for input and ensures it is one of the valid options.
 *
 * This function repeatedly prompts the user for input until a valid response is entered.
 * It takes a prompt message and a list of valid inputs, and it ensures that the user's input
 * matches one of the valid options. If the input is invalid, an error message is displayed
 * along with the list of valid options.
 *
 * @param prompt       The message displayed to the user when asking for input.
 * @param valid_inputs A vector of strings representing the valid inputs.
 * @return std::string The valid input entered by the user.
 */
std::string getValidInput(const std::string& prompt, const std::vector<std::string>& valid_inputs)
{
    std::unordered_set<std::string> valid_set(valid_inputs.begin(), valid_inputs.end());
    std::string answer;

    do
    {
        std::cout << prompt;
        std::cin >> answer;

        if (!valid_set.contains(answer))
        {
            std::cout << "Invalid input! Please enter one of the following: ";
            for (const auto& input : valid_inputs)
                std::cout << input << " ";

            std::cout << "\n";
        }

    } while (!valid_set.contains(answer));

    return answer;
}

/**
 * @brief Extracts a region of interest (ROI) from the image, ensuring the ROI is within image boundaries.
 *
 * This function checks if the specified ROI is within the boundaries of the given image.
 * If the ROI is completely within the image boundaries, it extracts and returns the corresponding sub-image.
 * If the ROI extends beyond the image boundaries, the function pads the image using reflection padding to
 * ensure the entire ROI can be safely extracted. The ROI coordinates are adjusted accordingly to the new padded image.
 *
 * @param img The input image from which the ROI is to be extracted.
 * @param roi The rectangle defining the region of interest. This parameter is updated if padding is applied.
 * @return cv::Mat The extracted ROI, which may be from the original or the padded image.
 */
cv::Mat getBoundarySafeROI(const cv::Mat& img, cv::Rect& roi)
{
    // Check if the ROI is within the image boundaries
    if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= img.cols && roi.y + roi.height <= img.rows)
    {
        return img(roi);
    }

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