#include "pipeline.h"

#include "hog_features_extraction.h"
#include "utils.h"
#include "svm_inference_input.h"
#include "template_matching.h"
#include "kmeans.h"
#include "eigenplanes.h"
#include "python_script.h"


// =============================================================================
//                                Perform KMeans By Size
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
//                              Perform KMeans By Intensity
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

// =============================================================================
//                          Resize Images Across Clusters
// =============================================================================
void resizeImagesAcrossClusters()
{
    const auto output_folder_path = createDirectory(std::filesystem::path(SRC_DIR_PATH), "resized_clusters");

    std::vector<std::string> intensity_cluster_paths;
    listDirectories(std::filesystem::path(SRC_DIR_PATH) / "kmeans_by_intensity", intensity_cluster_paths);

    for (size_t i = 0; i < intensity_cluster_paths.size(); ++i)
        resizeImgsSingleCluster(intensity_cluster_paths[i], output_folder_path, i);
}

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


void classify(const std::string& testing_img_id)
{
    // Construct the file paths using the image ID
    const std::string image_path = std::string(TESTING_DATASET_PATH) + "/" + testing_img_id + ".jpg";
    const std::string yolo_boxes_path = std::string(TESTING_DATASET_PATH) + "/" + testing_img_id + ".txt";

    // Read the source image
    cv::Mat src_img = cv::imread(image_path);
    if (!src_img.data)
        std::cerr << "Error: Could not read " << testing_img_id << ".jpg file\n";

    cv::Mat src_img_gray;
    cv::cvtColor(src_img, src_img_gray, cv::COLOR_BGR2GRAY);
    // Read YOLO boxes
    std::vector<cv::Rect> yolo_boxes;
    try
    {
        yolo_boxes = readYoloBoxes(std::filesystem::path(yolo_boxes_path));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
    }

    // Perform template matching 
    std::vector<cv::Point> max_corr_points = templateMatching(src_img_gray);

    // Classify points by YOLO boxes
    std::vector<cv::Point> max_corr_points_inside_yolo;
    std::vector<cv::Point> max_corr_points_outside_yolo;
    classifyPointsByYoloBoxes(yolo_boxes, max_corr_points, max_corr_points_inside_yolo, max_corr_points_outside_yolo);

    // Associate YOLO boxes with ROIs
    std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> yoloBox_roi_pairs;
    yoloBox_roi_pairs = associateYoloBoxesWithRois(yolo_boxes, max_corr_points_inside_yolo);

    // Label ROIs based on IOU with YOLO boxes
    constexpr double iou_threshold = 0.1;
    std::vector<std::pair<cv::Rect, std::string>> rois_labels_for_points_inside_yolo;
    rois_labels_for_points_inside_yolo =  labelRoisWithMaxIouAboveThreshold(yoloBox_roi_pairs, iou_threshold);

    // Generate ROIs with labels from points outside YOLO boxes and label them as "FP"
    std::vector<std::pair<cv::Rect, std::string>> rois_labels_for_points_outside_yolo;
    rois_labels_for_points_outside_yolo = generateRoisWithLabelsFromPoints_Outside_Yolo(max_corr_points_outside_yolo);

    // Concatenate ROI-label pairs
    std::vector<std::pair<cv::Rect, std::string>> rois_label_pairs_all_points;
    rois_label_pairs_all_points = concatenateRoiLabelPairs(rois_labels_for_points_inside_yolo, rois_labels_for_points_outside_yolo);




    // Create output directory fot the classification results
    std::filesystem::path output_dir = createDirectory(std::filesystem::path(SRC_DIR_PATH), "classification");


	// Save ROI-label pairs to file 
    saveRoiLabelPairsToFile(rois_label_pairs_all_points, (output_dir / "roi_label_pairs.csv").string());
    // Extract ROIs
    std::vector<cv::Rect> rois = extractRois(rois_label_pairs_all_points);
    // HOG features extraction
    std::vector<std::vector<float>> vector_of_hog_features;
    vector_of_hog_features = hog_features_extraction(rois, src_img_gray);

    // Write HOG features to CSV file
    std::filesystem::path testing_samples_path = output_dir / "testing_samples.csv";
    writeHogFeaturesToCsv(vector_of_hog_features, testing_samples_path.string());


    // Write the testing_img_id to a file
    try 
    {
        std::filesystem::path img_id_path = output_dir / "testing_img_id.txt";
        std::ofstream img_id_file(img_id_path);
        if (!img_id_file) 
        {
            throw std::ios_base::failure("Error: Could not open file to write testing_img_id");
        }
        img_id_file << testing_img_id;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }


    
    cv::Mat src_img_drawing = src_img.clone();
    drawRectangles(src_img_drawing, rois_label_pairs_all_points);
    imshow("Detection result", src_img_drawing, true, 0.4f);
    cv::imwrite((output_dir / "result.png").string(), src_img_drawing);

}

// =============================================================================
//                                Evaluate Performance
// =============================================================================
void evaluatePerformance()
{
    configureAndRunPythonScript();
}


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
    {"Classification", "generateEigenplanes"},
    {"Performance_evaluation", "Classification"}
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
    {"Classification", []() {
        std::string testing_img_id;
        std::cout << "Please enter the testing image ID: ";
        std::cin >> testing_img_id;
        classify(testing_img_id);
        createCompletionFile("Classification");
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

  Classification
    - This step classifies a given testing image based on the 
      extracted HOG features. It involves reading the image, 
      performing template matching, classifying points by YOLO 
      boxes, extracting ROIs, and using an SVM model to classify 
      the ROIs.

  Performance_evaluation
    - This step evaluates the performance of the classification 
      by running a Python script. It checks the accuracy and 
      other performance metrics of the SVM model.

Options:
--------

  --help
    - Show this message and exit.

==============================================================
    )";
}