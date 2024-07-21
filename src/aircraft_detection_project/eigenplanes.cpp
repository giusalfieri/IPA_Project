#include "eigenplanes.h"
#include "utils.h"




cv::Mat createDataMatrix(const std::vector<cv::Mat>& images)
{
    cv::Mat data;
    for (const cv::Mat& img : images)
    {
        cv::Mat img_vector = img.reshape(1, 1); // Flatten the image to a single row
        data.push_back(img_vector);
    }
    return data;
}

// Compute the average plane from a vector of images using PCA
cv::Mat eigenPlanes(const std::vector<cv::Mat>& vec, cv::Size img_dims)
{
    // Convert all images to CV_64F and create the data matrix from the vector of images
    std::vector<cv::Mat> vec64f;
    for (const auto& img : vec)
    {
        cv::Mat img_64f;
        img.convertTo(img_64f, CV_64F);
        vec64f.push_back(img_64f.reshape(1, 1));
    }
    cv::Mat data = createDataMatrix(vec64f);

    // Calculate the mean plane and subtract it from all planes
    cv::Mat mean_plane;
    cv::reduce(data, mean_plane, 0, cv::REDUCE_AVG);               // Calculate the mean plane
    cv::Mat data_centered = data - cv::repeat(mean_plane, data.rows, 1); // Subtract mean plane from all planes

    // Perform PCA on the centered data matrix
    cv::PCA pca(data_centered, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.95);

    // Project each image onto the PCA space and compute the average projection
    cv::Mat avg_projection = cv::Mat::zeros(1, pca.eigenvectors.rows, CV_64F);
    for (size_t i = 0; i < vec.size(); i++)
    {
        cv::Mat vec_i = data_centered.row(i);
        cv::Mat projection = pca.project(vec_i);
        avg_projection += projection;
    }
    avg_projection /= static_cast<double>(vec.size());

    // Reshape the average projection back into the original image dimensions
    cv::Mat avgPlaneCentered = pca.backProject(avg_projection);

    // Ensure the mean plane is in CV_64F
    mean_plane.convertTo(mean_plane, CV_64F);

    // Add the mean plane back to the centered average plane to get the final average plane
    cv::Mat avg_plane = avgPlaneCentered + mean_plane;

    // Normalize the result to the range [0, 255] and convert to CV_8U
    cv::normalize(avg_plane, avg_plane, 0, 255, cv::NORM_MINMAX);
    avg_plane = avg_plane.reshape(1, img_dims.height);
    avg_plane.convertTo(avg_plane, CV_8U);

    return avg_plane;
}