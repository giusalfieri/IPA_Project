#include "pipeline.h"

#include "hog_features_extraction.h"
#include "utils.h"
#include "detection.h"
#include "template_matching.h"
#include "kmeans.h"
#include "eigenplanes.h"
#include "python_script.h"
#include "svm_training.h"
#include "template_extraction.h"






// =============================================================================
//                                Perform K-Means By Size
// =============================================================================
void performKMeansBySize()
{

    const auto extracted_templates_folder_path = std::filesystem::path(SRC_DIR_PATH) /"extracted_templates";

    std::vector<std::string> template_paths;
    globFiles(extracted_templates_folder_path.string(), "/*.png", template_paths);

    std::vector<cv::Mat> extracted_templates;
    readImages(template_paths, extracted_templates);

    constexpr int num_clusters_by_size = 5;
    const cv::Mat labels = kmeansBySize(extracted_templates, num_clusters_by_size);

    const auto kmean_by_size_folder_path = createDirectory(std::filesystem::path(SRC_DIR_PATH), "kmeans_by_size");

    std::vector<std::filesystem::path> clusters_by_size_paths;
    clusters_by_size_paths.reserve(num_clusters_by_size);

    for (int i = 0; i < num_clusters_by_size; ++i) 
        clusters_by_size_paths.push_back(createDirectory(kmean_by_size_folder_path, "Cluster_" + std::to_string(i)));

    saveClusteredImages(extracted_templates, template_paths, labels, clusters_by_size_paths);
}
// =============================================================================



// =============================================================================
//                              Perform K-Means By Intensity
// =============================================================================
void performKMeansByIntensity()
{

    const auto kmeans_intensity_folder_path = createDirectory(std::filesystem::path(SRC_DIR_PATH),"kmeans_by_intensity");

    std::vector<std::string> clusters_by_size_paths;
    listDirectories(std::filesystem::path(SRC_DIR_PATH) / "kmeans_by_size", clusters_by_size_paths);

    constexpr int num_clusters_by_intensity = 6;

    for (size_t k = 0; k < clusters_by_size_paths.size(); k++)
    {
        std::vector<std::string> clustered_by_size_templates_path;
        globFiles(clusters_by_size_paths[k], "/*.png", clustered_by_size_templates_path);

        std::vector<cv::Mat> clustered_by_size_templates;
        readImages(clustered_by_size_templates_path, clustered_by_size_templates, cv::IMREAD_GRAYSCALE);

        cv::Mat labels_intensity_clusters = kmeansByIntensity(clustered_by_size_templates, num_clusters_by_intensity);
        const auto kmeans_intensity_cluster_group_path = createDirectory(kmeans_intensity_folder_path, "Group_" + std::to_string(k));

        std::vector<std::filesystem::path> clusters_by_intensity_paths;
        clusters_by_intensity_paths.reserve(num_clusters_by_intensity); 

        for (int j = 0; j < num_clusters_by_intensity; ++j) 
            clusters_by_intensity_paths.push_back(createDirectory(kmeans_intensity_cluster_group_path, "Cluster_By_Intensity_" + std::to_string(j)));

        saveClusteredImages(clustered_by_size_templates,clustered_by_size_templates_path,labels_intensity_clusters, clusters_by_intensity_paths);
    }
}
// =============================================================================



// =============================================================================
//                          Resize Images Across Clusters
// =============================================================================
// Resize Images in Single Cluster
void resizeImgsSingleCluster(const std::string& cluster_input_path, const std::filesystem::path& output_base_path, size_t cluster_index)
{
    std::vector<std::string> image_paths;
    globFiles(cluster_input_path, "/*.png", image_paths);

    std::vector<cv::Mat> images;
    readImages(image_paths, images, cv::IMREAD_GRAYSCALE);

    reshape2sameDim(images, calculateAvgDims(cluster_input_path));

    const auto cluster_output_path = createDirectory(output_base_path, "Cluster_same_size_" + std::to_string(cluster_index));

    for (size_t j = 0; j < images.size(); ++j)
    {
        auto image_stem = std::filesystem::path(image_paths[j]).stem();
        auto output_image_path = cluster_output_path / image_stem;
        cv::imwrite(output_image_path.string() + ".png", images[j]);
    }
}
// Resize Images Across Clusters
void resizeImagesAcrossClusters()
{
    const auto output_folder_path = createDirectory(std::filesystem::path(SRC_DIR_PATH), "resized_clusters");

    std::vector<std::string> intensity_cluster_paths;
    listDirectories(std::filesystem::path(SRC_DIR_PATH) / "kmeans_by_intensity", intensity_cluster_paths);

    for (size_t i = 0; i < intensity_cluster_paths.size(); ++i)
        resizeImgsSingleCluster(intensity_cluster_paths[i], output_folder_path, i);
}
// =============================================================================



// =============================================================================
//                                Generate Eigenplanes
// =============================================================================
void generateEigenplanes()
{
    const auto resized_clusters_dir_path = std::filesystem::path(SRC_DIR_PATH) / "resized_clusters";
    std::vector<std::string> single_resized_cluster_dir_paths;

    for (const auto& entry : std::filesystem::directory_iterator(resized_clusters_dir_path))
    {
        if (entry.is_directory()) 
            single_resized_cluster_dir_paths.push_back(entry.path().string());
    }

    const auto avg_airplanes_dir = createDirectory(std::filesystem::path(SRC_DIR_PATH),"avg_airplanes");

    for (size_t i = 0; i < single_resized_cluster_dir_paths.size(); i++)
    {
        std::vector<std::string> img_paths_in_single_resized_cluster;
        cv::glob(single_resized_cluster_dir_paths[i] + "/*.png", img_paths_in_single_resized_cluster);

        std::vector<cv::Mat> intensities_img;
        readImages(img_paths_in_single_resized_cluster, intensities_img, cv::IMREAD_GRAYSCALE);

        cv::Mat avg_airplane = eigenPlanes(intensities_img, calculateAvgDims(single_resized_cluster_dir_paths[i]));
        cv::imwrite((avg_airplanes_dir / ("avg_airplane" + std::to_string(i) + ".png")).string(), avg_airplane);
    }
}
// =============================================================================



// =============================================================================
//                                Airplanes Detection
// =============================================================================
void airplanesDetection(const std::string& testing_img_id)
{
    cv::Mat src_img, src_img_gray;
    std::vector<cv::Rect> yolo_boxes;

    readImageAndAnnotations(testing_img_id, src_img, src_img_gray, yolo_boxes);

    // Perform template matching 
    std::vector<cv::Point> max_corr_points = templateMatching(src_img_gray);

    auto rois_label_pairs_all_points = classifyAndLabelRois(yolo_boxes, max_corr_points);

   
    const std::filesystem::path output_dir = createDirectory(std::filesystem::path(SRC_DIR_PATH), "detection");

    saveResults(testing_img_id, rois_label_pairs_all_points, src_img_gray, output_dir);

    drawAndSaveResults(src_img, rois_label_pairs_all_points, output_dir);
}
// =============================================================================



// =============================================================================
//                                Evaluate Performance
// =============================================================================
void evaluatePerformance()
{
    configureAndRunPythonScript();
}
// =============================================================================





// =============================================================================
//  
// =============================================================================



// Path for step completion files
const std::filesystem::path stepStatePath = std::filesystem::path(SRC_DIR_PATH)/ "steps_completed";

// Helper function to create completion file
void createCompletionFile(const std::string& step)
{
    std::filesystem::create_directories(stepStatePath);
    std::ofstream(stepStatePath / (step + ".done")).close();
}

// Define valid steps and their dependencies
const std::unordered_map<std::string, std::string> stepDependencies = {
    {"extractTemplates", "training_phase"},
    {"KMeansBySize", "extractTemplates"},
    {"KMeansByIntensity", "KMeansBySize"},
    {"resizeImagesInClusters", "KMeansByIntensity"},
    {"generateEigenplanes", "resizeImagesInClusters"},
    {"Detection", "generateEigenplanes"},
    {"Performance_evaluation", "Detection"}
};

// Map for steps and their corresponding functions
std::unordered_map<std::string, std::function<void()>> stepFunctions = {
    {"training_phase", []() {
        extract_csv_for_svm_training();
        createCompletionFile("training_phase");
    }},
    {"extractTemplates", []() {
        extractTemplates();
        createCompletionFile("extractTemplates");
    }},
    {"KMeansBySize", []() {
        performKMeansBySize();
        createCompletionFile("KMeansBySize");
    }},
    {"KMeansByIntensity", []() {
        performKMeansByIntensity();
        createCompletionFile("KMeansByIntensity");
    }},
    {"resizeImagesInClusters", []() {
        resizeImagesAcrossClusters();
        createCompletionFile("resizeImagesInClusters");
    }},
    {"generateEigenplanes", []() {
        generateEigenplanes();
        createCompletionFile("generateEigenplanes");
    }},
    {"Detection", []() {
        std::string testing_img_id;
        std::cout << "Please enter the testing image ID: ";
        std::cin >> testing_img_id;
        airplanesDetection(testing_img_id);
        createCompletionFile("Detection");
    }},
    {"Performance_evaluation", []() {
        evaluatePerformance();
        createCompletionFile("Performance_evaluation");
    }},
    {"--help", printHelp}
};


void checkPreviousStep(const std::string& current_step)
{
    auto it = stepDependencies.find(current_step);
    if (it != stepDependencies.end())
    {
        const auto& previous_step = it->second;
        auto previous_step_file = stepStatePath / (previous_step + ".done");
        if (!std::filesystem::exists(previous_step_file))
            throw std::runtime_error("The step " + previous_step + " has not been executed yet. Cannot execute " + current_step + ".");
    }
}


void parseArguments(int argc, char** argv, std::vector<std::string>& steps)
{
    for (int i = 1; i < argc; ++i)
        steps.emplace_back(argv[i]);
}


void executeStep(const std::string& step)
{
    auto it = stepFunctions.find(step);
    if (it != stepFunctions.end())
    {
        it->second();
    }
    else
    {
        std::cerr << "\nUnknown step: " << step << "\n";
        printHelp();
    }
}


void printHelp()
{
    std::cout << R"(
==============================================================
               Aircraft Detection Project - HELP
==============================================================

Usage: program_name [step or option]

Steps:
------

  training_phase
    - This step initiates the training phase for an SVM model 
      by extracting data from a CSV file. The extracted data is 
      prepared and formatted to be used for training the SVM 
      algorithm.

  extractTemplates
    - This step involves extracting templates from the dataset. 
      Templates are essential parts of the images which will be 
      used for further processing and analysis.

  KMeansBySize
    - This step applies the K-Means clustering algorithm to 
      group data points (extracted templates) based on their 
      size. This helps in categorizing the templates into 
      clusters with similar dimensions.

  KMeansByIntensity
    - This step uses the K-Means clustering algorithm to group 
      data points (templates) based on their intensity levels. 
      Each cluster will contain templates with similar intensity 
      values.

  resizeImagesInClusters
    - This step involves resizing images within each cluster so 
      that all images in a cluster have the same dimensions. This 
      is necessary for further steps like generating eigenplanes.

  generateEigenplanes
    - This step generates eigenplanes for each cluster of images 
      that have been resized to the same dimensions. Eigenplanes 
      are used in various computer vision tasks to capture 
      essential features of the images.

  Detection
    - This step classifies a given testing image based on the 
      extracted HOG features. It involves reading the image, 
      performing template matching, classifying points by YOLO 
      boxes, extracting ROIs, and using an SVM model to classify 
      the ROIs.

  Performance_evaluation
    - This step evaluates the performance of the detection 
      by running a Python script. It checks the accuracy and 
      other performance metrics of the SVM model.

Options:
--------

  --help
    - Show this message and exit.

==============================================================
    )";
}
