#include "template_matching.h"

#include "utils.h"
#include <future>



/**
 * @brief Loads average plane images from a specified directory.
 *
 * This function loads all PNG images from a directory named "avg_airplanes" located at the source directory path (SRC_DIR_PATH).
 * It reads each image and stores it in a vector of cv::Mat objects. Only images that are successfully read are added to the vector.
 *
 * @return std::vector<cv::Mat> A vector of cv::Mat objects containing the loaded average plane images.
 */
std::vector<cv::Mat> loadAvgPlanes()
{
    const std::filesystem::path avg_airplanes_dir(std::filesystem::path(SRC_DIR_PATH) / "avg_airplanes");
    std::vector<std::string> avg_airplanes_paths;
    globFiles(avg_airplanes_dir.string(), "/*.png", avg_airplanes_paths);

    std::vector<cv::Mat> avg_planes;
    for (const auto& path : avg_airplanes_paths)
    {
        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.data)
            avg_planes.push_back(img);
    }
    return avg_planes;
}

/**
 * @brief Creates a range of angles.
 *
 * This helper function generates a vector of integers representing a range of angles.
 * The range starts from the specified start angle, increments by the specified step,
 * and stops before reaching the specified end angle.
 *
 * @param start The starting angle of the range (inclusive).
 * @param end The ending angle of the range (exclusive).
 * @param step The step size between consecutive angles.
 * @return std::vector<int> A vector of integers representing the range of angles.
 */
std::vector<int> angle_range(int start, int end, int step)
{
    std::vector<int> angles;
    for (int angle = start; angle < end; angle += step) 
        angles.push_back(angle);
    
    return angles;
}

/**
 * @brief Rotates an image by a specified angle.
 *
 * This function rotates a given image by the specified angle (in degrees) around its center.
 * The resulting image is adjusted to fit the entire rotated content without clipping.
 *
 * @param src_img The source cv::Mat object representing the image to be rotated.
 * @param degree_angle The angle (in degrees) by which the image should be rotated. Positive values rotate the image counter-clockwise.
 * @return cv::Mat A cv::Mat object representing the rotated image.
 */
cv::Mat rotateImage(const cv::Mat& src_img, int degree_angle)
{
    cv::Point rot_center = cv::Point(src_img.cols / 2.0f, src_img.rows / 2.0f);
    cv::Mat rotation_mat = cv::getRotationMatrix2D(rot_center, degree_angle, 1);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src_img.size(), degree_angle).boundingRect2f();
    rotation_mat.at<double>(0, 2) += bbox.width / 2.0f - rot_center.x;
    rotation_mat.at<double>(1, 2) += bbox.height / 2.0f - rot_center.y;

    cv::Mat dst;
    cv::warpAffine(src_img, dst, rotation_mat, bbox.size());
    return dst;
}


/**
 * @brief Transforms a point using the inverse of a given affine transformation matrix.
 *
 * This function takes a point and an affine transformation matrix, computes the inverse of the
 * affine transformation, and applies it to the given point. The transformed point is then returned.
 *
 * @param match_center The cv::Point object representing the point to be transformed.
 * @param rotation_mat The cv::Mat object representing the affine transformation matrix.
 * @return cv::Point The transformed cv::Point object.
 */
cv::Point transformPoint(const cv::Point& match_center, const cv::Mat& rotation_mat)
{
    cv::Mat inv_rotation_mat;
    cv::invertAffineTransform(rotation_mat, inv_rotation_mat);

    std::vector<cv::Point2f> pointsOriginal(1);
    cv::transform(std::vector<cv::Point2f>{match_center}, pointsOriginal, inv_rotation_mat);

    return pointsOriginal[0];
}



/**
 * @brief Perform template matching on a rotated source image and return the matched points.
 *
 * This function rotates the source image by a specified angle, performs template matching using the
 * provided template (avg_plane), and returns the coordinates of the best match points.
 *
 * @param src_img The source image in which to search for the template.
 * @param avg_plane The template image to be matched in the source image.
 * @param degree_angle The angle (in degrees) by which to rotate the source image before matching.
 * @return std::vector<cv::Point> A vector of points indicating the best match positions in the original source image coordinates.
 *
 * The function operates as follows:
 * 1. Rotates the source image by the specified degree angle.
 * 2. Performs normalized cross-correlation (NCC) template matching.
 * 3. Finds the location of the maximum NCC value, which indicates the best match.
 * 4. Converts the matched point coordinates from the rotated image back to the original image coordinates.
 * 5. Returns the matched points in the original image coordinates.
 */
std::vector<cv::Point> performTemplateMatching(const cv::Mat& src_img, const cv::Mat& avg_plane, int degree_angle)
{
    std::vector<cv::Point> local_matched_points;

    cv::Mat rotation_mat = cv::getRotationMatrix2D(cv::Point(src_img.cols / 2.0f, src_img.rows / 2.0f), degree_angle, 1);
    cv::Mat rotated_img = rotateImage(src_img, degree_angle);

    cv::Mat NCC_Output;
    cv::matchTemplate(rotated_img, avg_plane, NCC_Output, cv::TM_CCOEFF_NORMED);

    double maxVal;
    cv::Point maxP;
    cv::minMaxLoc(NCC_Output, nullptr, &maxVal, nullptr, &maxP);

    cv::Point matchCenter(maxP.x + avg_plane.cols / 2, maxP.y + avg_plane.rows / 2);
    cv::Point matchCenterOriginal = transformPoint(matchCenter, rotation_mat);

    local_matched_points.push_back(matchCenterOriginal);

    return local_matched_points;
}


/**
 * @brief Find template matches in the source image using multiple threads.
 *
 * This function performs template matching for multiple templates (avg_planes) across various rotation angles
 * on the source image using multiple threads to speed up the computation. It returns the matched points
 * for all templates and angles.
 *
 * @param src_img The source image in which to search for the templates.
 * @param avg_planes A vector of template images to be matched in the source image.
 * @return std::vector<cv::Point> A vector of points indicating the best match positions for all templates and angles in the original source image coordinates.
 *
 * The function operates as follows:
 * 1. Iterates over each template in avg_planes.
 * 2. For each template, iterates over angles from 0 to 360 degrees in steps defined by angle_step.
 * 3. For each angle, launches a new asynchronous task to perform template matching using the performTemplateMatching function.
 * 4. Collects the results from all asynchronous tasks.
 * 5. Combines all matched points from the asynchronous tasks into a single vector.
 * 6. Returns the combined vector of matched points.
 */
std::vector<cv::Point> matchTemplateMultiThreaded(const cv::Mat& src_img, const std::vector<cv::Mat>& avg_planes)
{
    std::vector<cv::Point> matched_points;
    constexpr auto angle_step = 5;
    std::vector<std::shared_future<std::vector<cv::Point>>> futures;

    for (const auto& avg_plane : avg_planes)
    {
        for (auto degree_angle : angle_range(0, 360, angle_step))
        {
            futures.emplace_back(std::async(std::launch::async, [src_img, avg_plane, degree_angle]() {
                return performTemplateMatching(src_img, avg_plane, degree_angle);
                }));
        }
    }

    for (auto& future : futures)
    {
        auto local_points = future.get();
        matched_points.insert(matched_points.end(), local_points.begin(), local_points.end());
    }

    return matched_points;
}

/**
 * @brief Perform template matching on the source image using multiple templates and multiple threads.
 *
 * This function loads a set of average template images and performs template matching on the source image
 * using multiple threads to handle different rotations and templates concurrently. It returns the points
 * in the source image where the templates best match.
 *
 * @param src_img The source image in which to search for the templates.
 * @return std::vector<cv::Point> A vector of points indicating the best match positions for all templates and angles in the original source image coordinates.
 *
 * The function operates as follows:
 * 1. Loads a set of average template images by calling the loadAvgPlanes function.
 * 2. Calls the matchTemplateMultiThreaded function to perform template matching for each template at various rotation angles using multiple threads.
 * 3. Returns the combined vector of matched points from all templates and angles.
 */
std::vector<cv::Point> templateMatching(const cv::Mat& src_img)
{
    // Load average planes
    std::vector<cv::Mat> avg_planes = loadAvgPlanes();

    // Find template matches
    std::vector<cv::Point> matched_points = matchTemplateMultiThreaded(src_img, avg_planes);

    return matched_points;
}