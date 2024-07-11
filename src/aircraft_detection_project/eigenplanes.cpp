#include "eigenplanes.h"

#include "utils.h"



// Create a data matrix from a vector of images
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


cv::Mat eigenPlanes(const std::vector<cv::Mat>& vec, cv::Size img_dims)
{
    // Step 1: Convert all images to CV_64F and create the data matrix from the vector of images
    std::vector<cv::Mat> vec64f;
    for (const auto& img : vec)
    {
        cv::Mat img_64f;
        img.convertTo(img_64f, CV_64F);
        vec64f.push_back(img_64f.reshape(1, 1));
    }
    cv::Mat data = createDataMatrix(vec64f);

    // Step 2: Calculate the mean face and subtract it from all faces
    cv::Mat mean_plane;
    cv::reduce(data, mean_plane, 0, cv::REDUCE_AVG); // Calculate the mean face
    cv::Mat data_centered = data - cv::repeat(mean_plane, data.rows, 1); // Subtract mean face from all faces

    // Step 3: Perform PCA on the centered data matrix
    cv::PCA pca(data_centered, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.95);

    // Step 4: Project each image onto the PCA space and compute the average projection
    cv::Mat avg_projection = cv::Mat::zeros(1, pca.eigenvectors.rows, CV_64F);
    for (size_t i = 0; i < vec.size(); i++)
    {
        cv::Mat vec_i = data_centered.row(i);
        cv::Mat projection = pca.project(vec_i);
        avg_projection += projection;
    }
    avg_projection /= static_cast<double>(vec.size());

    // Step 5: Reshape the average projection back into the original image dimensions
    cv::Mat avgPlaneCentered = pca.backProject(avg_projection);

    // Ensure the meanPlane is in CV_64F
    mean_plane.convertTo(mean_plane, CV_64F);

    // Add the mean face back to the centered average face to get the final average face
    cv::Mat avg_plane = avgPlaneCentered + mean_plane;

    // Normalize the result to the range [0, 255] and convert to CV_8U
    cv::normalize(avg_plane, avg_plane, 0, 255, cv::NORM_MINMAX);
    avg_plane = avg_plane.reshape(1, img_dims.height);
    avg_plane.convertTo(avg_plane, CV_8U);

    return avg_plane;
}