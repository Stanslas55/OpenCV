#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

void affichage(cv::Mat image, std::string titre);
cv::Mat lecture(const std::string filename, int flag);

void donnees_images_grises(cv::Mat image);
void donnees_images_couleur(cv::Mat image);

cv::Mat filtre_gris(cv::Mat image, cv::Mat filtre);
cv::Mat filtre_couleur(cv::Mat image, cv::Mat filtre);
cv::Mat demander_filtre();

void visage(cv::Mat& img, cv::CascadeClassifier& cascade, double scale);

void test_video(cv::Mat filtre);

int entree_valide(std::string entree, float* choix);
bool check_number(std::string str);

enum raccourcis
{
	gris = cv::IMREAD_GRAYSCALE,
	couleur = cv::IMREAD_COLOR,	
	video
};

enum couleurs
{
	B = 0,
	G = 1,
	R = 2
};

int main(int argc, char **argv)
{
	const std::string filename = "image.jpg";
	//const std::string filename = "2A.jpg"; // Image avec visage.
	cv::Mat image;
	//cv::CascadeClassifier cascade;
	int flag = video;
	double scale = 1;

	if (flag == video)
	{
		image = lecture(filename, gris);
	}
	else {
		image = lecture(filename, flag);
	}
	// Load the cascade classifier.	
	//cascade.load("../../haarcascade_frontalcatface.xml");

	//visage(image, cascade, scale);



	cv::Mat filtre;

	//affichage(image, "Mon image");
	if (flag == gris)
	{
		//donnees_images_grises(image);
		filtre = demander_filtre();
		affichage(image, "Mon image sans filtre");
		image = filtre_gris(image, filtre);
		affichage(image, "Mon image filtrée");
	}
	else if (flag == couleur)
	{
		//donnees_images_couleur(image);
		filtre = demander_filtre();
		affichage(image, "Mon image sans filtre");
		image = filtre_couleur(image, filtre);
		affichage(image, "Mon image filtrée");
	}
	else if (flag == video)
	{
		filtre = demander_filtre();
		test_video(filtre);
	}
	return EXIT_SUCCESS;
}

cv::Mat lecture(const std::string filename, int flag)
{
	cv::Mat image = cv::imread(filename, flag);
	if (image.empty())
	{
		std::cout << "Impossible d'ouvir l'image" << std::endl;
		return cv::Mat();
	}
	return image;
}

void affichage(cv::Mat image, std::string titre)
{		
	cv::namedWindow(titre, CV_WINDOW_NORMAL);
	cv::imshow(titre, image);
	cv::waitKey(0);
	cv::destroyWindow(titre);	
}

void donnees_images_grises(cv::Mat image)
{
	size_t i_lignes, i_colonnes, i_vector, i_occurences; // Les indices.
	size_t l_lignes = image.rows, l_colonnes = image.cols; // Les limites de boucles.
	unsigned int valeurGrise = 0, moyenneGrise = 0;
	unsigned int valeur = 0, occurences[256] = { 0 };

	std::vector<unsigned int> pixels_gris;

	std::cout << "\nSize : " << l_lignes << "*" << l_colonnes;
	 
	for (i_lignes = 0; i_lignes < l_lignes; i_lignes++)
	{
		for (i_colonnes = 0; i_colonnes < l_colonnes; i_colonnes++)
		{	
				valeurGrise = image.at<unsigned char>(i_lignes, i_colonnes);
				moyenneGrise += valeurGrise;
				pixels_gris.push_back(valeurGrise);
		}	
	} 
	
	moyenneGrise = moyenneGrise / (l_lignes * l_colonnes);
	std::sort(pixels_gris.begin(), pixels_gris.end());
	size_t resolution = l_lignes * l_colonnes;
	for (i_vector = 0; i_vector < resolution; i_vector++)
	{		
		if (valeur != pixels_gris[i_vector]) {
			valeur++;
		}
		occurences[valeur]++;
	}	
	std::cout << "\nMoyenne des valeurs grises : " << moyenneGrise << "\nMediane : " << pixels_gris[l_lignes*l_colonnes / 2]<< std::endl;

	// Vérification qu'on a bien tous les pixels.
	unsigned int compteur = 0, min = 9999999, max = 0;
	for (i_occurences = 0; i_occurences < 256; i_occurences++)
	{		
		compteur += occurences[i_occurences];	
		if (occurences[i_occurences] > max) max = occurences[i_occurences];
		if (occurences[i_occurences] < min) min = occurences[i_occurences];
	}

	for (i_occurences = 0; i_occurences < 256; i_occurences++)
	{		
		occurences[i_occurences] = (uint)((float)((occurences[i_occurences] - min) * 256 / (max - min)));	
	}
	cv::Mat Histogram(256, 256, CV_8UC1, cv::Scalar(0));

	for (i_colonnes = 0; i_colonnes < 256; i_colonnes++)
	{
		for (i_lignes = 0; i_lignes < 256; i_lignes++)
		{
			if (i_lignes < occurences[i_colonnes]) {
				Histogram.at<unsigned char>(255- i_lignes, i_colonnes) = 255;
			}		
		}
	}
	affichage(Histogram, "Mon Histogramme");
}

void donnees_images_couleur(cv::Mat image)
{
	//affichage(image, "Test");
	size_t i_lignes, i_colonnes, i_vector, i_occurences; // Les indices.
	size_t l_lignes = image.rows, l_colonnes = image.cols; // Les limites de boucles.
	size_t resolution = l_lignes * l_colonnes;

	unsigned char valeur[3] = { 0 };
	unsigned int occurences[3][256] = { 0 };

	std::vector<unsigned char> couleur[3];

	std::cout << "\nSize : " << l_lignes << "*" << l_colonnes;
	for (i_lignes = 0; i_lignes < l_lignes; i_lignes++)
	{
		for (i_colonnes = 0; i_colonnes < l_colonnes; i_colonnes++)
		{			
			couleur[B].push_back(image.at<cv::Vec3b>(i_lignes, i_colonnes)[B]);
			couleur[G].push_back(image.at<cv::Vec3b>(i_lignes, i_colonnes)[G]);
			couleur[R].push_back(image.at<cv::Vec3b>(i_lignes, i_colonnes)[R]);
		}
	}
	std::sort(couleur[B].begin(), couleur[B].end());
	std::sort(couleur[G].begin(), couleur[G].end());
	std::sort(couleur[R].begin(), couleur[R].end());

	for (i_vector = 0; i_vector < resolution; i_vector++)
	{
		if (valeur[B] != couleur[B][i_vector]) {
			valeur[B]++;
		}
		occurences[B][valeur[B]]++;

		if (valeur[G] != couleur[G][i_vector]) {
			valeur[G]++;
		}
		occurences[G][valeur[G]]++;

		if (valeur[R] != couleur[R][i_vector]) {
			valeur[R]++;
		}
		occurences[R][valeur[R]]++;
	}

	// Vérification qu'on a bien tous les pixels.
	unsigned int compteur[3] = { 0 };
	unsigned int min[3] = { 9999999 }, max[3] = { 0 };
	for (i_occurences = 0; i_occurences < 256; i_occurences++)
	{
		compteur[B] += occurences[B][i_occurences];
		compteur[G] += occurences[G][i_occurences];
		compteur[R] += occurences[R][i_occurences];
		if (occurences[B][i_occurences] > max[B]) max[B] = occurences[B][i_occurences];
		if (occurences[B][i_occurences] < min[B]) min[B] = occurences[B][i_occurences];

		if (occurences[G][i_occurences] > max[G]) max[G] = occurences[G][i_occurences];
		if (occurences[G][i_occurences] < min[G]) min[G] = occurences[G][i_occurences];

		if (occurences[R][i_occurences] > max[R]) max[R] = occurences[R][i_occurences];
		if (occurences[R][i_occurences] < min[R]) min[R] = occurences[R][i_occurences];
	}

	for (i_occurences = 0; i_occurences < 256; i_occurences++)
	{
		occurences[B][i_occurences] = (uint)((float)((occurences[B][i_occurences] - min[B]) * 256 / (max[B] - min[B])));
		occurences[G][i_occurences] = (uint)((float)((occurences[G][i_occurences] - min[G]) * 256 / (max[G] - min[G])));
		occurences[R][i_occurences] = (uint)((float)((occurences[R][i_occurences] - min[R]) * 256 / (max[R] - min[R])));
	}
	cv::Mat HistogramB(256, 256, CV_8UC1, cv::Scalar(0));
	cv::Mat HistogramG(256, 256, CV_8UC1, cv::Scalar(0));
	cv::Mat HistogramR(256, 256, CV_8UC1, cv::Scalar(0));

	for (i_colonnes = 0; i_colonnes < 256; i_colonnes++)
	{
		for (i_lignes = 0; i_lignes < 256; i_lignes++)
		{
			if (i_lignes < occurences[B][i_colonnes]) {
				HistogramB.at<unsigned char>(255 - i_lignes, i_colonnes) = 255;
			}
			if (i_lignes < occurences[G][i_colonnes]) {
				HistogramG.at<unsigned char>(255 - i_lignes, i_colonnes) = 255;
			}
			if (i_lignes < occurences[R][i_colonnes]) {
				HistogramR.at<unsigned char>(255 - i_lignes, i_colonnes) = 255;
			}
		}
	}
	affichage(HistogramB, "Mon Histogramme Bleu");
	affichage(HistogramG, "Mon Histogramme Vert");
	affichage(HistogramR, "Mon Histogramme Rouge");
}

cv::Mat filtre_gris(cv::Mat image, cv::Mat filtre)
{
	int ordre = filtre.rows;
	int indice_lignes, indice_colonnes;
	int image_rows = image.rows, image_cols = image.cols;
	/*
	std::cout << "taille de l'image : " << image_rows << " * " << image_cols << std::endl;
	std::cout << "taille du filtre : " << ordre << " * " << ordre << std::endl;	

	for (indice_lignes = 0; indice_lignes < ordre; indice_lignes++)
	{
		for (indice_colonnes = 0; indice_colonnes < ordre; indice_colonnes++)
		{
			std::cout << "[" << filtre.at<float>(indice_lignes, indice_colonnes) << "]";
		}
		std::cout << std::endl;
	}
	*/
	// Il faut créer une nouvelle image qui sera l'image filtrée.
	cv::Mat image_filtre(image_rows, image_cols, CV_32SC1, cv::Scalar(255));


	// Pour l'application du filtre, on parcours l'image de 0 + (ordre - 1) / 2 à (limite_ligne - (ordre -1) / 2)
	int debut = (int)((float)(ordre - 1) / 2);
	int fin_lignes = image_rows - (int)((float)(ordre - 1) / 2) - 1;
	int fin_colonnes = image_cols - (int)((float)(ordre - 1) / 2) - 1;
	int decallage_lignes, decallage_colonnes;
	int somme = 0;

	float tmp_val;
	uchar tmp;

	for (indice_lignes = debut; indice_lignes < fin_lignes; indice_lignes++)
	{
		for (indice_colonnes = debut; indice_colonnes < fin_colonnes; indice_colonnes++)
		{
			somme = 0;
			for (decallage_lignes = -debut; decallage_lignes < debut + 1; decallage_lignes++)
			{				
				for (decallage_colonnes = -debut; decallage_colonnes < debut + 1; decallage_colonnes++)
				{
					tmp_val = filtre.at<float>(debut + decallage_lignes, debut + decallage_colonnes);					
					tmp = image.at<uchar>(indice_lignes + decallage_lignes, indice_colonnes + decallage_colonnes);
					
					somme += tmp_val * (int)tmp;
				}
			}			
			image_filtre.at<int>(indice_lignes, indice_colonnes) = somme;
		}		
	}	
	// Pour résoudre le bug de la Bibliothèque.
	image_filtre.convertTo(image_filtre, CV_8U, 255); // [0..1] -> [0..255] range
	return image_filtre;
}

cv::Mat filtre_couleur(cv::Mat image, cv::Mat filtre)
{
	int ordre = filtre.rows;
	int indice_lignes, indice_colonnes;
	int image_rows = image.rows, image_cols = image.cols;
/*
	std::cout << "taille de l'image : " << image_rows << " * " << image_cols << std::endl;
	std::cout << "taille du filtre : " << ordre << " * " << ordre << std::endl;

	for (indice_lignes = 0; indice_lignes < ordre; indice_lignes++)
	{
		for (indice_colonnes = 0; indice_colonnes < ordre; indice_colonnes++)
		{
			std::cout << "[" << filtre.at<float>(indice_lignes, indice_colonnes) << "]";
		}
		std::cout << std::endl;
	}
*/
	// Il faut créer une nouvelle image qui sera l'image filtrée.
	cv::Mat image_filtre(image_rows, image_cols, CV_8UC3, cv::Scalar(255, 255, 255)); // Une image en couleur.

	// Pour l'application du filtre, on parcours l'image de 0 + (ordre - 1) / 2 à (limite_ligne - (ordre -1) / 2)
	int debut = (int)((float)(ordre - 1) / 2);
	//int fin_lignes = image_rows - (int)((float)(ordre - 1) / 2) - 1;
	//int fin_colonnes = image_cols - (int)((float)(ordre - 1) / 2) - 1;
	int decallage_lignes, decallage_colonnes;
	float somme[3] = { 0 };

	float tmp_val;
	cv::Vec3b tmp;	

	for (indice_lignes = 0; indice_lignes < image_rows; indice_lignes++)
	//for (indice_lignes = debut; indice_lignes < fin_lignes; indice_lignes++)
	{
		for (indice_colonnes = 0; indice_colonnes < image_cols; indice_colonnes++)
		//for (indice_colonnes = debut; indice_colonnes < fin_colonnes; indice_colonnes++)
		{
			somme[B] = 0;
			somme[G] = 0;
			somme[R] = 0;

			for (decallage_lignes = -debut; decallage_lignes < debut + 1; decallage_lignes++)
			{
				for (decallage_colonnes = -debut; decallage_colonnes < debut + 1; decallage_colonnes++)
				{
					tmp_val = filtre.at<float>(debut + decallage_lignes, debut + decallage_colonnes);
					// Quand je ne gérais pas les bords.
					//tmp = image.at<cv::Vec3b>(indice_lignes + decallage_lignes, indice_colonnes + decallage_colonnes); 					
				
					// Gestion des bords avec un padding nul.
					if ((indice_lignes - debut + decallage_lignes) < 0 || (indice_colonnes - debut + decallage_colonnes) < 0 || (indice_lignes + debut + decallage_lignes) > image_rows - 1 || (indice_colonnes + debut + decallage_colonnes) > image_cols - 1)
					{
						tmp[B] = 0;
						tmp[G] = 0;					
						tmp[R] = 0;
					}
					else
					{						
						tmp = image.at<cv::Vec3b>(indice_lignes + decallage_lignes, indice_colonnes + decallage_colonnes);
					}
					somme[B] += tmp_val * tmp[B];
					somme[G] += tmp_val * tmp[G];
					somme[R] += tmp_val * tmp[R];
				}
			}
			
			somme[B] = (somme[B] < 0) ? 0 : somme[B];
			somme[G] = (somme[G] < 0) ? 0 : somme[G];
			somme[R] = (somme[R] < 0) ? 0 : somme[R];

			somme[B] = (somme[B] > 255) ? 255 : somme[B];
			somme[G] = (somme[G] > 255) ? 255 : somme[G];
			somme[R] = (somme[R] > 255) ? 255 : somme[R];

			image_filtre.at<cv::Vec3b>(indice_lignes, indice_colonnes)[B] = somme[B];
			image_filtre.at<cv::Vec3b>(indice_lignes, indice_colonnes)[G] = somme[G];
			image_filtre.at<cv::Vec3b>(indice_lignes, indice_colonnes)[R] = somme[R];
		}
	}		
	return image_filtre;
}

cv::Mat demander_filtre()
{
	float ordre = -1;
	float coeff = -1;
	std::string entree = "";
	int indice_lignes, indice_colonnes;
	
	do {
		std::cout << "Entrer l'ordre du filtre\n> ";
		std::cin >> entree;
	} while (!entree_valide(entree, &ordre) || ((int)ordre % 2 == 0) || ordre < 3);
	std::cout << "ordre: " << ordre << std::endl;
	ordre = (int)ordre; //A enlever ?
	cv::Mat filtre(ordre, ordre, CV_32FC1, cv::Scalar(0));

	for (indice_lignes = 0; indice_lignes < ordre; indice_lignes++)
	{
		for (indice_colonnes = 0; indice_colonnes < ordre; indice_colonnes++)
		{
			std::cout << "[" << filtre.at<int>(indice_colonnes, indice_lignes) << "]";
		}
		std::cout << std::endl;
	}
	std::cout << "\nEntrer les coefficients" << std::endl;
	for (indice_lignes = 0; indice_lignes < ordre; indice_lignes++)
	{
		std::cout << "\nLigne " << indice_lignes + 1 << std::endl;
		for (indice_colonnes = 0; indice_colonnes < ordre; indice_colonnes++)
		{			
			// Ajouter la récupération du coefficient et l'ajouter au bon endroit dans la matrice.
			do {		
				std::cout << "\tColonne " << indice_colonnes + 1 << " > ";
				std::cin >> entree;
			} while (!entree_valide(entree, &coeff));
			//std::cout << "coeff entre: " << coeff << std::endl;
			filtre.at<float>(indice_colonnes, indice_lignes) = coeff;
		}		
	}
	std::cout << std::endl;
	return filtre;
}

void visage(cv::Mat& image, cv::CascadeClassifier& cascade, double scale)
{
	size_t indice;
	std::vector<cv::Rect> faces;
	
	cv::Rect r;
	cv::Scalar color;
	//cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	cascade.detectMultiScale(image, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	for (indice = 0; indice < faces.size(); indice++)
	{
		r = faces[indice];
		color = cv::Scalar(255, 0, 0);
		rectangle(image, cvPoint(cvRound(r.x * scale), cvRound(r.y * scale)), cvPoint(cvRound((r.x +
			r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)), color, 3, 8, 0);
	}
	imshow("Visage", image);
}

void test_video(cv::Mat filtre)
{
	cv::VideoCapture cam(0); // Je capture ma camera.
	cv::Mat frame;
	char c;
	int frame_width, frame_height;

	if (!cam.isOpened())
	{
		std::cout << "Erreur Camera." << std::endl;
		return;
	}

	frame_width = cam.get(CV_CAP_PROP_FRAME_WIDTH);
	frame_height = cam.get(CV_CAP_PROP_FRAME_HEIGHT);

	//cv::VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 1, cv::Size(frame_width, frame_height), true);

	while (1)
	{		
		cam >> frame;
		if (frame.empty()) break;

		//video.write(frame);

		// Pas exploitable.
		//frame = filtre_gris(frame, filtre); 

		frame = filtre_couleur(frame, filtre);
		cv::imshow("Webcam", frame);
		
		c = (char)cv::waitKey(1);
		if (c == 27) // Si on appuis sur ECHAP.
		{
			break;
		}
	}
	cam.release();
	//video.release();
	cv::destroyAllWindows();
	return;
}

int entree_valide(std::string entree, float* choix)
{
	if (check_number(entree)) {
		*choix = std::stof(entree);
		return 1;
	}
	else {
		std::cout << "Vous avez entre une lettre !! L'entree n'est pas enregistree..." << std::endl;
		return 0;
	}
}

bool check_number(std::string str) {
	for (unsigned int i = 0; i < str.length(); i++)
		if (isdigit(str[i]) == false && str[i] != '-' && str[i] != '.')
			return false;
	return true;
}