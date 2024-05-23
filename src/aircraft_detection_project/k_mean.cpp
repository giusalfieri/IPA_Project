#include "k_mean.h"
#include "utils.h"


void kMeansClustering_BySize()
{
	std::filesystem::path kmeans_folder_path = createDirectory(std::filesystem::path(DATASET_PATH).parent_path(), "kmeans_by_size");

	const auto templates_extracted_path = std::filesystem::path(DATASET_PATH).parent_path() / "extracted_templates";
	const auto templates_paths_pattern = templates_extracted_path.string() + std::string("/*.png");
	std::vector<std::string> template_paths;
	cv::glob(templates_paths_pattern, template_paths);


	std::vector<cv::Mat> templates;
	for (const auto& path : template_paths)
	{
		cv::Mat img = cv::imread(path);
		if (img.data)
			templates.push_back(img);
	}


	// Creation of a matrix of templates.size() rows and 2 columns
	// This matrix will store the dimensions (width, height) of each extracted template
	cv::Mat templates_dims(templates.size(), 2, CV_32F);


	// Fill the sizes matrix with the dimensions of each extracted template
	for (int i = 0; i < templates.size(); i++)
	{
		float* yRow = templates_dims.ptr<float>(i);
		yRow[0] = templates[i].cols;  // the first column stores the width  of the i-th template
		yRow[1] = templates[i].rows;  // the first column stores the height of the i-th template
	}


	// K-Means Clustering 
	cv::Mat labels, centers;
	int K = 6;
	cv::kmeans(
		templates_dims,
		K,
		labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
		50,
		cv::KMEANS_PP_CENTERS,
		centers
	);



	//DA MIGLIORARE
	for (int i = 0; i < K; i++)
	{
	
		std::filesystem::path new_cluster_path = createDirectory(kmeans_folder_path, "Cluster_" + std::to_string(i) );

		for (int j = 0; j < labels.rows; j++)
		{
			if (labels.at<int>(j) == i)
			{
				// Definisco la cartella nella quale si vanno a salvare i templates
				std::string kmean_template_name = "template_" + std::to_string(j) + ".png";

				std::filesystem::path save_path = new_cluster_path / kmean_template_name;

				//Le immagini di input non vengono alterate e quindi labels e images hanno lo stesso indice
				//Utilizzo il metodo string per convertire il percorso in una stringa
				cv::imwrite(save_path.string(), templates[j]);
			}
		}
	

	}

}

void kMeansClustering_ByIntensity()
{
  
    fs::path dataset_path(DATASET_PATH);
    
    // Percorso dove verranno salvati i nuovi cluster basati sull'intensità
    fs::path kmeans_folder_path = fs::path(dataset_path).parent_path() / "kmeans_by_intensity";
    fs::create_directory(kmeans_folder_path);

    std::vector<std::string> image_files;
    std::vector<cv::Mat> images;

	int num_clusters = 15;
    int attempts = 200; // Numero di tentativi per il clustering



  // Creo le cartelle per salvare i cluster successivamente
    for (int i = 0; i < num_clusters; i++) 
    {
        std::string directory_name = "Cluster_" + std::to_string(i);
        fs::path new_cluster_path = kmeans_folder_path / directory_name;
        fs::create_directory(new_cluster_path);
    }

    // Itera su ciascuna delle k cartelle di cluster basate sulle dimensioni
	//Da migliorare in modo tale da renderlo uguale alle dimensioni del clustering by size
    for (int k = 0; k < 6; k++)
    {
        // Definisco la cartella dalla quale si vanno a prendere i templates
        fs::path kmean_size_path = dataset_path.parent_path() / fs::path("kmeans_by_size") / fs::path("Cluster_" + std::to_string(k));
        
        // Carica i percorsi delle immagini dalla cartella del cluster corrente
        for (const auto &k_mean_size_file : fs::directory_iterator(kmean_size_path))
        {
            if (k_mean_size_file.is_regular_file())
            {
                image_files.push_back(k_mean_size_file.path().string());
            }
        }
    }

    // Andiamo a caricare tutte le immagini presenti nella cartella corrente del k_means size leggendole dal vettore image_files
    for (const auto &file : image_files)
    {
        cv::Mat image = cv::imread(file);
        if (!image.empty())
        {
            images.push_back(image);
        }
	//For debug purposes
        else
        {
            std::cerr << "Warning: Image " << file << " could not be read." << std::endl;
        }
    }

    std::cout << "Number of images loaded: " << images.size() << std::endl;

    // Crea una matrice per le intensità
    cv::Mat intensities(images.size(), 1, CV_32F);
    for (int i = 0; i < images.size(); i++) 
    {
        // Sommo l'intensità calcolata canale per canale e poi effettuo la media
        cv::Scalar mean_intensity = cv::mean(images[i]);
        float intensity = (mean_intensity[0] + mean_intensity[1] + mean_intensity[2]) / 3.0f;
        intensities.at<float>(i, 0) = intensity;
    }

    cv::Mat labels, centers;    
    
    cv::kmeans(
        intensities, 
        num_clusters, 
        labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.3), 
        attempts, 
        cv::KMEANS_PP_CENTERS,
        centers
    );

    // Sfrutto l'ordinamento delle labels per salvare le immagini
    int saved_images_count = 0;
    for (int i = 0; i < labels.rows; i++) 
    {
        int cluster_idx = labels.at<int>(i);
        std::string file_name = "image_" + std::to_string(i) + ".png";
        fs::path save_path = kmeans_folder_path / ("Cluster_" + std::to_string(cluster_idx)) / file_name;
        if (cv::imwrite(save_path.string(), images[i]))
        {
            saved_images_count++;
        }
		//For debug purposes
        else
        {
            std::cerr << "Error: Could not save image " << file_name << std::endl;
        }
    }
	//For debug purposes
    std::cout << "Number of images saved: " << saved_images_count << std::endl;
}


