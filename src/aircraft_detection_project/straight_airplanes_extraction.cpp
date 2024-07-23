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


/**
 * @brief Extracts, processes, and saves straightened airplane images from a dataset.
 *
 * This function processes a dataset of images and their corresponding YOLO label files to extract, straighten,
 * and save airplane images. It performs the following steps:
 * 1. Reads the paths of the dataset images and YOLO label files.
 * 2. Creates a directory to save the straightened airplane images.
 * 3. Iterates through each dataset image:
 *    a. Reads the image and converts it to HSV color space.
 *    b. Processes the YOLO labels to obtain bounding boxes for the airplanes.
 *    c. Prompts the user to select and optionally rotate the airplanes.
 *    d. Binarizes the selected airplane regions.
 *    e. Calculates geometric moments to determine the orientation of the airplanes.
 *    f. Extracts and rotates the airplane images to be upright.
 *    g. Saves the processed airplane images to the specified directory.
 *
 * @note This function assumes that the dataset images and YOLO label files are in the same directory.
 * @note The processed images are saved in the `straight_airplanes` directory within the source directory.
 *
 * @see globFiles
 * @see createDirectory
 * @see processYoloLabels
 * @see selectAirplanes
 * @see binarizeAirplanes
 * @see calculateGeometricMoments
 * @see extractRotatedAirplanes
 * @see saveStraightAirplanes
 */
void extractStraightAirplanes()
{
    std::vector<std::string> dataset_img_paths;
    globFiles(std::string(TRAINING_DATASET_PATH) + "/" + "dataset_for_straight_airplanes_extraction", "/*.jpg", dataset_img_paths);

    std::vector<std::string> yolo_labels_paths;
    globFiles(std::string(TRAINING_DATASET_PATH) + "/" + "dataset_for_straight_airplanes_extraction", "/*.txt", yolo_labels_paths);

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





/**
 * @brief Selects and processes airplane images based on user input.
 *
 * This function iterates through a list of YOLO bounding boxes, displays each corresponding airplane image to the user,
 * and prompts the user to decide whether to process and select the airplane. The user can also choose to rotate and save the airplane image.
 *
 * @param[in] img The input image containing the airplanes.
 * @param[in] yolo_boxes A vector of `cv::Rect` objects representing the YOLO bounding boxes for each airplane.
 * @param[out] selected_airplanes_yolo_boxes A vector of `cv::Rect` objects to store the selected YOLO bounding boxes for further processing.
 * @param[in,out] count An integer counter used to generate unique filenames for the saved images. It is incremented with each saved image.
 * @param[in] img_filename The original filename of the image, used as part of the new filename.
 * @param[in] straight_airplanes_folder The path to the directory where the processed images will be saved.
 *
 * @note The function displays each airplane image and prompts the user to decide whether to process it.
 * @note If the user chooses to process an airplane, it is added to the `selected_airplanes_yolo_boxes` vector.
 * @note If the user chooses to rotate and save an airplane, it is rotated by a multiple of 90 degrees clockwise and saved.
 *
 * @see getValidInput
 * @see rotate90
 * @see saveAirplane
 * @see cv::imshow
 * @see cv::waitKey
 * @see cv::Rect
 */
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
 * @brief Binarizes regions of interest (ROIs) in an image using adaptive thresholding.
 *
 * This function extracts specified ROIs from an image channel, applies adaptive thresholding to binarize them,
 * and performs morphological dilation to enhance the binary images. The resulting binary images are stored in an output vector.
 *
 * @param[in] channel The input image channel from which the ROIs are extracted.
 * @param[in] selected_airplanes_yolo_boxes A vector of `cv::Rect` objects representing the ROIs to be binarized.
 * @param[out] bin_airplanes A vector of `cv::Mat` objects to store the resulting binary images.
 *
 * @note The function adjusts the block size for adaptive thresholding to ensure it is odd.
 * @note A constant `C` is used to fine-tune the thresholding.
 * @note Morphological dilation is applied using an elliptical structuring element.
 *
 * @see cv::adaptiveThreshold
 * @see cv::morphologyEx
 * @see cv::getStructuringElement
 * @see cv::THRESH_BINARY
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
 * @brief Calculates geometric moments for a set of binary airplane images and corresponding YOLO bounding boxes.
 *
 * This function computes the geometric moments for each binary airplane image, calculates the center and orientation angle,
 * and adjusts the center based on the corresponding YOLO bounding box. The results are stored in a vector of pairs, where
 * each pair contains the corrected center point and the orientation angle in degrees.
 *
 * @param[in] bin_airplanes A vector of binary `cv::Mat` objects representing the segmented airplanes.
 * @param[in] yolo_boxes A vector of `cv::Rect` objects representing the YOLO bounding boxes for each airplane.
 * @param[out] geometric_moments_descriptors A vector of pairs where each pair contains the corrected center point
 *            (`cv::Point2f`) and the orientation angle (`double`) in degrees for each airplane.
 *
 * @note The function assumes that the contours are sorted by area in descending order to select the largest contour for moment calculation.
 * @note The orientation angle is corrected to degrees from radians.
 *
 * @see cv::findContours
 * @see cv::moments
 * @see std::atan2
 * @see rad2degrees
 * @see sortByDescendingArea
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

/**
 * @brief Corrects the orientation angle of an airplane image.
 *
 * This function adjusts the orientation angle of an airplane image to ensure it is within a specific range.
 * If the angle is between 0 and 90 degrees, it adds 90 degrees to the angle.
 * If the angle is negative, it adjusts the angle to be within the 0-180 degree range.
 *
 * @param[in] angle The original orientation angle of the airplane in degrees.
 * @return The corrected orientation angle in degrees.
 */
double correctAngle(double angle)
{
    if (angle >= 0 && angle <= 90)
        return angle + 90.0f;
    else 
        return 90.0f - (-angle);
}

/**
 * @brief Calculates a centered region of interest (ROI) around a given point with a specified size and scale factor.
 *
 * This function computes a rectangular ROI centered at the given point, with dimensions scaled by the specified scale factor.
 *
 * @param[in] center The center point around which the ROI is calculated.
 * @param[in] size The size of the ROI before scaling.
 * @param[in] scale_factor The factor by which the size of the ROI is scaled.
 * @return A `cv::Rect` representing the calculated ROI.
 *
 * @note The resulting ROI may need to be adjusted to ensure it stays within the image boundaries when used.
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
 * @brief Extracts and rotates airplane images from the input image based on geometric moments and YOLO bounding boxes.
 *
 * This function processes binary airplane images and their corresponding geometric moments and YOLO bounding boxes
 * to extract, rotate, and resize the airplanes. The resulting straightened airplane images are stored in the output vector.
 *
 * The steps are as follows:
 * 1. Iterate through each binary airplane image.
 * 2. Check if the current index is valid.
 * 3. Calculate the center and correct the angle using geometric moments.
 * 4. Define the region of interest (ROI) and get a safe ROI within image boundaries.
 * 5. Rotate the ROI to align the airplane.
 * 6. Define the final ROI and extract the straightened airplane image.
 * 7. Store the straightened airplane image in the output vector if the final ROI is valid.
 *
 * @param[in] img The input image from which the airplanes are extracted.
 * @param[in] bin_airplanes A vector of binary `cv::Mat` objects representing the segmented airplanes.
 * @param[in] geometric_moments_descriptors A vector of pairs containing the center points and angles (in degrees) for each airplane.
 * @param[in] yolo_boxes A vector of `cv::Rect` objects representing the YOLO bounding boxes for each airplane.
 * @param[out] airplanes_vector A vector of `cv::Mat` objects to store the extracted and rotated airplane images.
 *
 * @note The function uses scaling factors to adjust the size of the ROI for better extraction.
 * @note If the final ROI is out of image boundaries, the airplane is skipped.
 *
 * @see correctAngle
 * @see calculateCenteredROI
 * @see getBoundarySafeROI
 * @see cv::getRotationMatrix2D
 * @see cv::warpAffine
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



/**
 * @brief Displays and optionally saves a list of airplane images after user confirmation and optional rotation.
 *
 * This function iterates through a list of airplane images, displaying each one to the user.
 * The user can choose to save each image, optionally rotating it before saving.
 * The saved images are stored in a specified directory with unique filenames.
 *
 * @param[in] airplanes A vector of `cv::Mat` objects representing the airplane images to be processed.
 * @param[in,out] count An integer counter used to generate unique filenames for the saved images.
 *                      It is incremented with each saved image.
 * @param[in] img_filename The original filename of the image, used as part of the new filename.
 * @param[in] straight_airplanes_folder The path to the directory where the images will be saved.
 *
 * @note The function displays each image and prompts the user to decide whether to keep it.
 *       If the user chooses to keep an image, they can also choose to rotate it before saving.
 *
 * @see getValidInput
 * @see rotate90
 * @see cv::imshow
 * @see cv::waitKey
 * @see cv::imwrite
 */
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
 * @brief Saves an airplane image to a specified directory with a unique filename.
 *
 * This function saves the provided airplane image to a specified directory.
 * The filename is generated using a count and the original image filename.
 *
 * @param[in] airplane The `cv::Mat` object representing the airplane image to be saved.
 * @param[in,out] count An integer counter used to generate unique filenames for the saved images.
 *                      It is incremented with each call to the function.
 * @param[in] img_filename The original filename of the image, used as part of the new filename.
 * @param[in] straight_airplane_folder The path to the directory where the image will be saved.
 *
 * @see cv::imwrite
 * @see std::filesystem::path
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
 * @brief Prompts the user for input until a valid input is provided.
 *
 * This function displays a prompt to the user and reads the user's input.
 * It continues to prompt the user until a valid input from the provided list is entered.
 *
 * @param[in] prompt The message displayed to the user when asking for input.
 * @param[in] valid_inputs A vector of strings representing the valid inputs.
 * @return A string containing the valid input entered by the user.
 *
 * @note The function uses an unordered set to store the valid inputs for efficient lookup.
 *
 * @see std::unordered_set
 * @see std::vector
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
 * @brief Extracts a region of interest (ROI) from an image, ensuring it stays within image boundaries.
 *
 * This function checks if the specified ROI is within the boundaries of the given image.
 * If the ROI is within boundaries, it returns the corresponding sub-image.
 * If the ROI exceeds the image boundaries, it pads the image using reflection border, adjusts the ROI, and then extracts the sub-image.
 *
 * @param[in] img The input image from which the ROI is extracted.
 * @param[in,out] roi The region of interest specified as a `cv::Rect`. If the ROI exceeds the image boundaries,
 *                it is adjusted to fit within the padded image.
 * @return A `cv::Mat` containing the extracted sub-image corresponding to the ROI.
 *
 * @see cv::copyMakeBorder
 * @see cv::Rect
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
