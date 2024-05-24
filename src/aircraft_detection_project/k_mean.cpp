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
	int K = 3;
	cv::kmeans(
		templates_dims,
		K,
		labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
		50,
		cv::KMEANS_PP_CENTERS,
		centers
	);



	for (int i = 0; i < K; i++)
	{

		
		std::filesystem::path new_cluster_path = createDirectory(kmeans_folder_path, "Cluster_" + std::to_string(i) );

		for (int j = 0; j < templates.size(); j++)
		{
			if (labels.at<int>(j) == i)
			{
				// Definisco la cartella nella quale si vanno a salvare i templates
				//std::string kmean_template_name = "template_" + std::to_string(j) + ".png";
				//std::cout << std::filesystem::path(template_paths[j]).stem().string() << "\n";
				//std::cout << j << "\n";
				auto kmean_template_name = std::filesystem::path(template_paths[j]).stem().string();

				std::filesystem::path clustered_template_path(new_cluster_path/kmean_template_name);

				//Le immagini di input non vengono alterate e quindi labels e images hanno lo stesso indice
				//Utilizzo il metodo string per convertire il percorso in una stringa
				cv::imwrite(clustered_template_path.string(), templates[j]);
			}
		}
	}

}



void kMeansClustering_ByIntensity()
{
    std::filesystem::path dataset_path(DATASET_PATH);
    
    // Percorso dove verranno salvati i nuovi cluster basati sull'intensità
    std::filesystem::path kmeans_intensity_folder_path = dataset_path.parent_path() / "kmeans_by_intensity";
    std::filesystem::create_directory(kmeans_intensity_folder_path);

    int NUM_CLUSTERS_KMEANS_SIZE = 3;
    int NUM_CLUSTERS_KMEANS_INTENSITY = 5;
    int attempts = 100; // Numero di tentativi per il clustering

    // Itera su ciascuna delle k cartelle di cluster basate sulle dimensioni
    //Da migliorare in modo tale da renderlo uguale alle dimensioni del clustering by size
    for (int k = 0; k < NUM_CLUSTERS_KMEANS_SIZE; k++)
    {
        std::vector<std::string> image_files;
        std::vector<cv::Mat> images;

        // Definisco la cartella dalla quale si vanno a prendere i templates
        std::filesystem::path kmean_size_path(dataset_path.parent_path() / fs::path("kmeans_by_size") / fs::path("Cluster_" + std::to_string(k)));

        // Vado a creare le cartelle in cui verranno salvati i risultati del kmeans per le intensità
        std::filesystem::path new_intensity_cluster_directory(kmeans_intensity_folder_path / fs::path("Intensity_cluster_" + std::to_string(k + 1)));
        std::filesystem::create_directory(new_intensity_cluster_directory);

        // For debug purposes
        std::cout << new_intensity_cluster_directory.string() << std::endl;

        // Carica i percorsi delle immagini dalla cartella del cluster per dimensione corrente
        for (const auto &k_mean_size_file : fs::directory_iterator(kmean_size_path))
        {
            if (k_mean_size_file.is_regular_file())
            {
                image_files.push_back(k_mean_size_file.path().string());
            }
        }

        // For debug purposes
        std::cout << image_files.size() << std::endl;

        // Andiamo a caricare tutte le immagini presenti nella cartella corrente del k_means size leggendole dal vettore image_files
        for (const auto &file : image_files)
        {
            // Vado a prendere il percorso salvato all'interno del vettore images_files e lo carico
            cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
            if (image.data)
            {
                images.push_back(image);
            }
            else
            {
                std::cerr << "Warning: Image " << file << " could not be read." << std::endl;
            }
            
        }

        // For debug purposes
        std::cout << "Number of images loaded: " << images.size() << std::endl;

        // Crea una matrice per le intensità
        cv::Mat intensities(images.size(), 1, CV_32F);

        for (int i = 0; i < images.size(); i++)
        {
            float* yRow = intensities.ptr<float>(i);
            // Sommo l'intensità calcolata canale per canale e poi effettuo la media
            cv::Scalar mean_intensity = cv::mean(images[i]);
            //float intensity = (mean_intensity[0] + mean_intensity[1] + mean_intensity[2]) / 3.0f;
            float intensity = static_cast <float> (mean_intensity[0]);
            yRow[0] =intensity;
        }

        cv::Mat labels, centers;

        cv::kmeans(
            intensities,
            NUM_CLUSTERS_KMEANS_INTENSITY,
            labels,
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1),
            attempts,
            cv::KMEANS_PP_CENTERS,
            centers
        );

        // Sfrutto l'ordinamento delle labels per salvare le immagini
        int saved_images_count = 0;

        // Creo le cartelle per i nuovi cluster di intensità
        for (int i = 0; i < NUM_CLUSTERS_KMEANS_INTENSITY; i++)
        {
            fs::path cluster_path = new_intensity_cluster_directory / ("Cluster_" + std::to_string(i + 1));
            fs::create_directory(cluster_path);
        }

        // Salvo le immagini nei rispettivi cluster
        for (int i = 0; i < labels.rows; i++)
        {
            int cluster_idx = labels.at<int>(i);
            std::string file_name = "image_" + std::to_string(i) + ".png";

            // Percorso di salvataggio
            std::filesystem::path save_path = new_intensity_cluster_directory / ("Cluster_" + std::to_string(cluster_idx + 1)) / file_name;

            // For debug purposes
            //std::cout << save_path.string() << std::endl;

            if (cv::imwrite(save_path.string(), images[i]))
            {
                saved_images_count++;
            }
            else
            {
                std::cerr << "Error: Could not save image " << file_name << std::endl;
            }
        }

        // For debug purposes
        std::cout << "Number of images saved: " << saved_images_count << std::endl;

        // Pulisce i vettori per il prossimo ciclo
        images.clear();
        image_files.clear();
    }
}

void resizeImages(const int num_clusters_by_size, const int num_clusters_by_intensity)
{

    std::filesystem::path dataset_path(DATASET_PATH);

    //(dataset parent folder) + kmeans_by_intensity + untenisty_cluster_i + cluster_i;
    for (int i = 0; i < num_clusters_by_size; i++)
    {
        for (int j = 0; j < num_clusters_by_intensity; j++)
        {

            int max_width = 0;
            int max_height = 0;

            std::filesystem::path cluster_path
                (dataset_path.parent_path() / 
                std::filesystem::path("kmeans_by_intensity")/ 
                std::filesystem::path("intensity_cluster_") / std::filesystem::path(std::to_string(i)) / 
                std::filesystem::path("cluster_") / 
                std::filesystem::path(std::to_string(j)
                ));

            std::vector<std::string>final_clusters_paths;
            const auto final_clusters_paths_pattern = cluster_path.string() + std::string("/*.png");
            cv::glob(final_clusters_paths_pattern, final_clusters_paths);

            std::vector<cv::Mat> final_clusters;
            for (const auto& path : final_clusters_paths)
            {
                cv::Mat img = cv::imread(path);
                if (img.data)
                    final_clusters.push_back(img);
            }

            for (const auto& img : final_clusters)
            {
                max_width  = std::max(max_width,  img.cols);
                max_height = std::max(max_height, img.rows);
            }

            // Apply mirroring to each image to make them reach the maximum size
            for (auto& img : final_clusters)
            {
                int padding_width = max_width - img.cols;
                int padding_height = max_height - img.rows;

                cv::Mat img_with_mirroring;
                cv::copyMakeBorder(img, img_with_mirroring, 0, padding_height, 0, padding_width, cv::BORDER_REFLECT); // Use mirroring

                img = img_with_mirroring;
            }

        }
    }

}
