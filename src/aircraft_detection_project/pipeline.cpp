#include "pipeline.h"

#include "hog_features_extraction.h"
#include "utils.h"
#include "template_matching.h"
#include "kmeans.h"
#include "eigenplanes.h"
#include "python_script.h"
#include "svm_training.h"
#include "straight_airplanes_extraction.h"





// =============================================================================
//                                Perform K-Means By Size
// =============================================================================
/**
 * @brief Performs K-Means clustering on extracted templates based on their size and saves the clustered images.
 *
 * This function reads images from a specified directory, performs K-Means clustering based on the dimensions
 * of the images, and saves the clustered images into corresponding directories.
 *
 * The steps are as follows:
 * 1. Reads image file paths from the specified directory.
 * 2. Loads the images into a vector of `cv::Mat`.
 * 3. Performs K-Means clustering based on the size of the images.
 * 4. Creates directories for each cluster.
 * 5. Saves the clustered images into the respective directories.
 *
 * @note This function assumes that the directory `SRC_DIR_PATH/straight_airplanes` exists and contains the images
 *       to be clustered.
 * @note The number of clusters is set to 5.
 *
 * @see globFiles
 * @see readImages
 * @see kmeansBySize
 * @see createDirectory
 * @see saveClusteredImages
 */
void performKMeansBySize()
{

    const auto extracted_templates_folder_path = std::filesystem::path(SRC_DIR_PATH) /"straight_airplanes";

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
/**
 * @brief Performs K-Means clustering on images based on their intensity and saves the clustered images.
 *
 * This function reads images that have been previously clustered by size, performs K-Means clustering
 * based on the intensity of the images, and saves the clustered images into corresponding directories.
 *
 * The steps are as follows:
 * 1. Creates a directory for saving the intensity-based clusters.
 * 2. Lists the directories of size-based clusters.
 * 3. For each size-based cluster:
 *    a. Reads the images in grayscale.
 *    b. Performs K-Means clustering based on the intensity of the images.
 *    c. Creates directories for each intensity-based cluster within the current size-based cluster.
 *    d. Saves the intensity-clustered images into the respective directories.
 *
 * @note This function assumes that the directory `SRC_DIR_PATH/kmeans_by_size` exists and contains the images
 *       that have been previously clustered by size.
 * @note The number of intensity-based clusters is set to 6.
 *
 * @see listDirectories
 * @see globFiles
 * @see readImages
 * @see kmeansByIntensity
 * @see createDirectory
 * @see saveClusteredImages
 */
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
/**
 * @brief Resizes images within a single cluster to the same dimensions and saves them.
 *
 * This function reads images from a specified input cluster directory, resizes them to
 * the same dimensions based on the average dimensions of the images, and saves the resized
 * images into a specified output directory.
 *
 * @param[in] cluster_input_path The path to the input directory containing the cluster images.
 * @param[in] output_base_path The base path to the output directory where resized images will be saved.
 * @param[in] cluster_index The index of the current cluster, used to name the output directory.
 *
 * @note This function assumes that the input directory contains images in `.png` format.
 *
 * @see globFiles
 * @see readImages
 * @see reshape2sameDim
 * @see calculateAvgDims
 * @see createDirectory
 * @see cv::imwrite
 */
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
/**
 * @brief Resizes images across multiple clusters to the same dimensions and saves them.
 *
 * This function lists the directories of intensity-based clusters, resizes images within
 * each cluster to the same dimensions, and saves the resized images into corresponding
 * directories within a base output directory.
 *
 * The steps are as follows:
 * 1. Creates the base output directory for resized clusters.
 * 2. Lists the directories containing intensity-based clusters.
 * 3. For each intensity-based cluster, resizes the images and saves them to the output directory.
 *
 * @note This function assumes that the directory `SRC_DIR_PATH/kmeans_by_intensity` exists and contains
 *       the images clustered by intensity.
 *
 * @see listDirectories
 * @see resizeImgsSingleCluster
 * @see createDirectory
 */
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
/**
 * @brief Generates average planes (eigenplanes) for clustered images and saves them.
 *
 * This function reads images from resized cluster directories, computes the average plane
 * for each cluster using PCA, and saves the resulting average planes into a specified directory.
 *
 * The steps are as follows:
 * 1. Lists the directories containing resized clusters.
 * 2. For each resized cluster:
 *    a. Reads the images in grayscale.
 *    b. Computes the average plane (eigenplane) using PCA.
 *    c. Saves the average plane image into the `avg_airplanes` directory.
 *
 * @note This function assumes that the directory `SRC_DIR_PATH/resized_clusters` exists and contains
 *       the images that have been resized and clustered.
 *
 * @see readImages
 * @see eigenPlanes
 * @see calculateAvgDims
 * @see createDirectory
 * @see cv::glob
 * @see cv::imwrite
 */
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
//                                Evaluate Performance
// =============================================================================
/**
 * @brief Evaluates the performance of the SVM model.
 *
 * This function runs a Python script to evaluate the performance of the SVM model.
 * The script calculates various performance metrics and displays the results.
 *
 * @see configureAndRunPythonScript
 */
void evaluatePerformance()
{
    configureAndRunPythonScript();
}
// =============================================================================





// Path for step completion files
const std::filesystem::path stepStatePath = std::filesystem::path(SRC_DIR_PATH)/ "steps_completed";

/**
 * @brief Creates a completion file for a specified step.
 *
 * This function creates a directory for step state files if it does not exist,
 * and then creates an empty file named "<step>.done" to indicate the completion of the specified step.
 *
 * @param[in] step The name of the step for which to create the completion file.
 */
void createCompletionFile(const std::string& step)
{
    std::filesystem::create_directories(stepStatePath);
    std::ofstream(stepStatePath / (step + ".done")).close();
}

/**
 * @brief Defines the dependencies between steps.
 *
 * This unordered map defines the dependencies between different steps in the process.
 * Each entry maps a step to its required preceding step.
 */
const std::unordered_map<std::string, std::string> stepDependencies = {
    {"KMeansBySize", "extractStraightAirplanes"},
    {"KMeansByIntensity", "KMeansBySize"},
    {"resizeImagesInClusters", "KMeansByIntensity"},
    {"generateEigenplanes", "resizeImagesInClusters"},
    {"extract_SVM_Training_Data", "generateEigenplanes"},
    {"Performance_evaluation", "extract_SVM_Training_Data"}
};

/**
 * @brief Maps step names to their corresponding functions.
 *
 * This unordered map defines the relationship between step names and the functions
 * that implement those steps. Each entry maps a step name to a function that performs
 * the step and creates a completion file upon successful execution.
 */
std::unordered_map<std::string, std::function<void()>> stepFunctions = {
    {"extractStraightAirplanes", []() {
        extractStraightAirplanes();
        createCompletionFile("extractStraightAirplanes");
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
    {"extract_SVM_Training_Data", []() {
        generateSvmTrainingData();
        createCompletionFile("extract_SVM_Training_Data");
    }},
    {"Performance_evaluation", []() {
        evaluatePerformance();
        createCompletionFile("Performance_evaluation");
    }},
    {"--help", printHelp}
};

/**
 * @brief Checks if the previous step required for the current step has been executed.
 *
 * This function verifies if the step that the current step depends on has been executed
 * by checking for the existence of a corresponding ".done" file. If the previous step has
 * not been executed, it throws a runtime error.
 *
 * @param[in] current_step The name of the current step to be executed.
 *
 * @throws std::runtime_error If the previous step required for the current step has not been executed.
 */
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

/**
 * @brief Parses command-line arguments to extract the list of steps to be executed.
 *
 * This function reads command-line arguments and stores them in a vector of steps.
 *
 * @param[in] argc The number of command-line arguments.
 * @param[in] argv The array of command-line argument strings.
 * @param[out] steps A vector of strings where the parsed steps will be stored.
 */
void parseArguments(int argc, char** argv, std::vector<std::string>& steps)
{
    for (int i = 1; i < argc; ++i)
        steps.emplace_back(argv[i]);
}


/**
 * @brief Executes the specified step if it is defined in the step functions map.
 *
 * This function looks up the specified step in the map of step functions and executes
 * the corresponding function if it is found. If the step is not found, it prints an error message
 * and displays the help information.
 *
 * @param[in] step The name of the step to be executed.
 *
 * @see printHelp
 */
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

/**
 * @brief Prints the help message for the Aircraft Detection Project.
 *
 * This function displays a detailed help message that includes usage instructions, descriptions
 * of the various steps in the project, and available options.
 */
void printHelp()
{
    std::cout << R"(
==============================================================
               Aircraft Detection Project - HELP
==============================================================

Usage: program_name [step or option]

Steps:
------

  extractStraightAirplanes
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

  extract_SVM_Training_Data
    - This step processes the dataset images and their 
      corresponding YOLO labels to generate training data for an 
      SVM. It performs template matching, classifies points, 
      extracts HOG features for true positives and false 
      positives, and saves the features to CSV files for SVM 
      training.

  Performance_evaluation
    - This step evaluates the performance of the SVM model by 
      running a Python script. It checks the accuracy and other 
      performance metrics of the model.

Options:
--------

  --help
    - Show this message and exit.

==============================================================
    )";
}