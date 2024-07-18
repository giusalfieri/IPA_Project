#include "template_matching.h"

#include <future>
#include "utils.h"


// Load average planes from the directory
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

// Helper function to create a range of angles
std::vector<int> angle_range(int start, int end, int step)
{
    std::vector<int> angles;
    for (int angle = start; angle < end; angle += step) 
        angles.push_back(angle);
    
    return angles;
}

// Rotate the image by a specified angle
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

// Apply inverse affine transformation to the matched point
cv::Point transformPoint(const cv::Point& match_center, const cv::Mat& rotation_mat)
{
    cv::Mat inv_rotation_mat;
    cv::invertAffineTransform(rotation_mat, inv_rotation_mat);

    std::vector<cv::Point2f> pointsOriginal(1);
    cv::transform(std::vector<cv::Point2f>{match_center}, pointsOriginal, inv_rotation_mat);

    return pointsOriginal[0];
}

// Perform template matching and return matched points
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

// Find template matches in the source image using multiple threads
std::vector<cv::Point> findTemplateMatchesMultiThreaded(const cv::Mat& src_img, const std::vector<cv::Mat>& avg_planes)
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


std::vector<cv::Point> templateMatching(const cv::Mat& src_img)
{
    // Load average planes
    std::vector<cv::Mat> avg_planes = loadAvgPlanes();
    // Find template matches
    std::vector<cv::Point> matched_points = findTemplateMatchesMultiThreaded(src_img, avg_planes);

    return matched_points;
}