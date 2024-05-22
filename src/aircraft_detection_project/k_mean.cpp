#include "k_mean.h"
#include "utils.h"


void kMeansClustering_BySize()
{
	fs::path dataset_path(DATASET_PATH);

	fs::path kmeans_path = dataset_path.parent_path();

	fs::path kmeans_folder_path = createDirectory(kmeans_path, "kmeans_by_size");

    std::vector<std::string> image_files;

	std::vector<cv::Mat> images;

	//Definisco la cartella dalla quale si vanno a prendere i templates
	fs::path templates_extracted_path = dataset_path.parent_path() / "extracted_templates";
	
	for (const auto &template_file : fs::directory_iterator(templates_extracted_path))
	{
		if (template_file.is_regular_file())
		{
			//Andiamo a riempire il vettore con i percorsi di ciascuna immagine
			image_files.push_back(template_file.path().string());
		}
	}
    

    for (const auto &file : image_files)
    {
		//Andiamo a caricare tutte le immagini presenti nella cartella "extracted_templates" leggendole dal vettore image_files
        cv::Mat image = cv::imread(file);
        if (!image.empty())
        {
            images.push_back(image);
        }
    }

    
	
	//Andiamo a creare una matrice di dimensione [righe] sizeof images vector X [colonne] 2 
	//Matrice di CV_32F come richiesto dalla funzione kmeans
	cv::Mat sizes(images.size(), 2, CV_32F);


	//Andiamo a riempire la matrice sizes con le dimensioni di ciascun template estratto
	for (int i = 0; i < images.size(); i++)
	{
		sizes.at<float>(i, 0) = images[i].cols;
		sizes.at<float>(i, 1) = images[i].rows;
	}


	cv::Mat labels, centers;
	int k = 4;	

	cv::kmeans(
		sizes, 
		k, 
		labels,
		cv::TermCriteria (cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 
		3, 
		cv::KMEANS_PP_CENTERS,
		centers
		);
    
	//Tramite questo ciclo for si vanno a visualizzare i templates appartenenti ad un certo cluster 
	for (int i = 0; i < k; i++)
    {
       // std::cout << "Cluster " << i << ":\n";
		std::string directory_name = "Cluster_" + std::to_string(i);
    	
		fs::path new_cluster_path = createDirectory(kmeans_folder_path, directory_name);

        for (int j = 0; j < labels.rows; j++)
        {
            if (labels.at<int>(j) == i)
            {
				//Per ogni cartella cluster vado a salvare gli elementi al suo interno
                std::string kmean_template_name = "template_" + std::to_string(j) + ".png";
                
				fs::path save_path = new_cluster_path / kmean_template_name;

				//Le immagini di input non vengono alterate e quindi labels e images hanno lo stesso indice
				//Utilizzo il metodo string per convertire il percorso in una stringa
                cv::imwrite(save_path.string(), images[j]);
            }
        }
        std::cout << "\n";


    }

}







void kMeansClustering_ByIntensity()
{
 // Kmean andrà effettuato poi sulla cartella ricomposta
    // Vado a richiamare il percorso della cartella contenente i templates dopo aver effettuato il kmean sulle dimensioni
    fs::path dataset_path(DATASET_PATH);
    fs::path kmeans_intensity = dataset_path.parent_path();
    
    fs::path kmeans_folder_path = fs::path(dataset_path).parent_path() / "kmeans_by_intensity";
    fs::create_directory(kmeans_folder_path);

    for (int k = 0; k < 4; k++)
    {
        std::vector<std::string> image_files;
        std::vector<cv::Mat> images;

        // Definisco la cartella dalla quale si vanno a prendere i templates
        fs::path kmean_size_path = dataset_path.parent_path() / fs::path("kmeans_by_size") / fs::path("Cluster_" + std::to_string(k));
        
        // For debug purposes
        // std::cout << kmean_size_path.string() << std::endl;

        for (const auto &k_mean_size_file : fs::directory_iterator(kmean_size_path))
        {
            if (k_mean_size_file.is_regular_file())
            {
                // Andiamo a riempire il vettore con i percorsi di ciascuna immagine
                image_files.push_back(k_mean_size_file.path().string());
            }
        }

        for (const auto &file : image_files)
        {
            // Andiamo a caricare tutte le immagini presenti nella cartella "extracted_templates" leggendole dal vettore image_files
            cv::Mat image = cv::imread(file);
            if (!image.empty())
            {
                images.push_back(image);
            }
            else
            {
                std::cerr << "Warning: Image " << file << " could not be read." << std::endl;
            }
        }

        std::cout << "Number of images loaded: " << images.size() << std::endl;

        cv::Mat intensities(images.size(), 1, CV_32F);
        for (int i = 0; i < images.size(); i++) 
        {
            cv::Scalar mean_intensity = cv::mean(images[i]);
            float intensity = (mean_intensity[0] + mean_intensity[1] + mean_intensity[2]) / 3.0f;
            intensities.at<float>(i, 0) = intensity;
        }
        
        cv::Mat labels, centers;    
        int num_clusters = 4; // Modifica questo valore in base al numero di immagini
        
        cv::kmeans(
            intensities, 
            num_clusters, 
            labels,
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 
            3, 
            cv::KMEANS_PP_CENTERS,
            centers
        );

        for (int i = 0; i < num_clusters; i++) 
        {
            std::string directory_name = "Cluster_" + std::to_string(i);
            fs::path new_cluster_path = kmeans_folder_path / directory_name;
            fs::create_directory(new_cluster_path);
        }

        for (int i = 0; i < labels.rows; i++) 
        {
            int cluster_idx = labels.at<int>(i);
            std::string file_name = "image_" + std::to_string(i) + ".png";
            fs::path save_path = kmeans_folder_path / ("Cluster_" + std::to_string(cluster_idx)) / file_name;
            cv::imwrite(save_path.string(), images[i]);
        }
    }



}



// float Get_Intensity(cv::Mat img)
// {


// 	//O si itera con un ciclo for sull'immagine

// 	// float intensity = 0;
//     // for (int i = 0; i < img.rows; i++)
//     // {
//     //     for (int j = 0; j < img.cols; j++)
//     //     {
//     //         intensity += img.at<uchar>(i, j);
//     //     }
//     // }
//     // return intensity/img.size();

// 	//Oppure si può utilizzare la funzione mean()
// 	if (img.channels() == 1)
//     {
// 		//Controllo solamente che l'immagine non sia in scala di grigi
//         cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
//     }
// 	cv::Scalar intensity_by_channel = cv::mean(img);
	
// 	float intensity_value = intensity_by_channel[0] + intensity_by_channel[1] + intensity_by_channel[2] / 3.0f;

// 	return intensity_value;

// }

// }


