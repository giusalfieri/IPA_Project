#include "template_matching.h"

#include "utils.h"
#include <future>


/**
 * @brief Loads average airplane images from the specified directory.
 *
 * This function reads all PNG images from the "avg_airplanes" directory within the source directory
 * and loads them into a vector of `cv::Mat` objects.
 *
 * @return A vector of `cv::Mat` objects containing the loaded average airplane images.
 *
 * @note The function assumes that the average airplane images are stored in the "avg_airplanes" directory
 *       within the source directory defined by `SRC_DIR_PATH`.
 * @note Only images that are successfully read are added to the vector.
 *
 * @see globFiles
 * @see cv::imread
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
 * @brief Generates a range of angles from start to end with a specified step.
 *
 * This function creates a vector of integers representing angles starting from `start`,
 * incremented by `step`, and ending before `end`.
 *
 * @param[in] start The starting angle.
 * @param[in] end The ending angle (exclusive).
 * @param[in] step The step size between consecutive angles.
 * @return A vector of integers representing the range of angles.
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
 * This function rotates the given image by a specified angle around its center.
 * It adjusts the bounding box to ensure the entire rotated image fits within the resulting image.
 *
 * @param[in] src_img The source image to be rotated.
 * @param[in] degree_angle The angle in degrees by which the image should be rotated.
 * @return A `cv::Mat` object containing the rotated image.
 *
 * @note The function uses the center of the image as the rotation point and adjusts the translation
 *       to ensure the entire rotated image fits within the new bounding box.
 *
 * @see cv::getRotationMatrix2D
 * @see cv::warpAffine
 * @see cv::RotatedRect
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
 * This function takes a point and an affine transformation matrix, computes the inverse of the matrix,
 * and applies it to the point to obtain its transformed coordinates.
 *
 * @param[in] match_center The point to be transformed.
 * @param[in] rotation_mat The affine transformation matrix to be inverted and applied to the point.
 * @return A `cv::Point` representing the transformed coordinates of the input point.
 *
 * @note The function uses `cv::invertAffineTransform` to compute the inverse of the affine transformation matrix.
 * @note The function uses `cv::transform` to apply the inverted matrix to the point.
 *
 * @see cv::invertAffineTransform
 * @see cv::transform
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
 * @brief Performs template matching on a source image with a rotated template.
 *
 * This function rotates the source image by a specified angle, performs template matching using the normalized cross-correlation method,
 * and returns the coordinates of the matched points transformed back to the original image coordinates.
 *
 * @param[in] src_img The source image in which to perform template matching.
 * @param[in] avg_plane The template image used for matching.
 * @param[in] degree_angle The angle in degrees by which to rotate the source image for matching.
 * @return A vector of `cv::Point` objects representing the coordinates of the matched points in the original image.
 *
 * @note The function uses `cv::getRotationMatrix2D` to compute the rotation matrix and `rotateImage` to rotate the source image.
 * @note The function uses `cv::matchTemplate` with the `cv::TM_CCOEFF_NORMED` method to perform template matching.
 * @note The function uses `transformPoint` to transform the coordinates of the matched points back to the original image coordinates.
 *
 * @see cv::getRotationMatrix2D
 * @see rotateImage
 * @see cv::matchTemplate
 * @see cv::minMaxLoc
 * @see transformPoint
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
 * @brief Performs multi-threaded template matching on a source image using multiple average planes.
 *
 * This function performs template matching on a source image using a set of average planes, rotating each plane by various angles.
 * It uses multi-threading to parallelize the matching process, combining the results into a single list of matched points.
 *
 * @param[in] src_img The source image in which to perform template matching.
 * @param[in] avg_planes A vector of `cv::Mat` objects representing the average planes used for matching.
 * @return A vector of `cv::Point` objects representing the coordinates of all matched points.
 *
 * @note The function uses a step of 5 degrees for rotating the average planes.
 * @note The function uses `std::async` with `std::launch::async` to perform template matching in parallel.
 * @note The function collects and combines the results from all threads.
 *
 * @see performTemplateMatching
 * @see angle_range
 * @see std::async
 * @see std::shared_future
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
 * @brief Performs template matching on a source image using pre-loaded average planes.
 *
 * This function loads a set of average planes and performs multi-threaded template matching on the source image.
 * It returns the coordinates of all matched points found in the image.
 *
 * @param[in] src_img The source image in which to perform template matching.
 * @return A vector of `cv::Point` objects representing the coordinates of all matched points.
 *
 * @note The function uses `loadAvgPlanes` to load the average planes from the predefined directory.
 * @note The function uses `matchTemplateMultiThreaded` to perform multi-threaded template matching.
 *
 * @see loadAvgPlanes
 * @see matchTemplateMultiThreaded
 */
std::vector<cv::Point> templateMatching(const cv::Mat& src_img)
{
    // Load average planes
    std::vector<cv::Mat> avg_planes = loadAvgPlanes();

    // Find template matches
    std::vector<cv::Point> matched_points = matchTemplateMultiThreaded(src_img, avg_planes);

    return matched_points;
}