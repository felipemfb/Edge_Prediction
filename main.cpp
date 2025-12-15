#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

float LAPLACE_GAUSS_MIN = 2.0;
float LAPLACE_GAUSS_MAX = 2.2;

int CANNY_LOW_THRESH = 0;
int CANNY_HIGH_THRESH = 350;

int SOBEL_THRESH = 225;

// Performance
float P(float faux_negatifs, float contours_corrects, float faux_positifs) { return(contours_corrects / (contours_corrects + faux_positifs + faux_negatifs)); }
// Le taux de faux positifs
float TFP(float faux_negatifs, float contours_corrects, float faux_positifs) { return (faux_positifs / (contours_corrects + faux_positifs + faux_negatifs)); }
// Le taux de faux négatifs
float TFN(float faux_negatifs, float contours_corrects, float faux_positifs) { return (faux_negatifs / (contours_corrects + faux_positifs + faux_negatifs)); }

void countContours(const cv::Mat& detected_in, const cv::Mat& reference_in) {
	using namespace std; using namespace cv;

	float faux_negatifs = 0, true_positifs = 0, faux_positifs = 0;

	Mat detected;
	if (detected_in.type() != CV_8U) detected_in.convertTo(detected, CV_8U);
	else detected = detected_in.clone();

	Mat reference;
	if (!reference_in.empty()) {
		if (reference_in.type() != CV_8U) reference_in.convertTo(reference, CV_8U);
		else reference = reference_in.clone();
	}
	else reference = Mat();

	if (reference.empty()) {
		cout << "Aucune image de référence..." << endl << endl;
		return;
	}

	if (detected.size() != reference.size()) {
		cerr << "ERREUR : Tailles différentes entre la valeur détectée et la valeur de référence" << endl;
		return;
	}

	Mat detectedBin, referenceBin;
	threshold(detected, detectedBin, 0, 255, THRESH_BINARY);
	threshold(reference, referenceBin, 0, 255, THRESH_BINARY);

	int rows = detectedBin.rows;
	int cols = detectedBin.cols;

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			bool d = detectedBin.at<uchar>(y, x) > 0;
			bool r = referenceBin.at<uchar>(y, x) > 0;
			if (!r) {
				if (d) {
					faux_positifs++;
				}
			}
			else {
				bool find = false;
				for (int dy = -1; dy <= 1 && !find; dy++) {
					for (int dx = -1; dx <= 1 && !find; dx++) {
						int Y = y + dy;
						int X = x + dx;
						if (Y >= 0 && Y < rows && X >= 0 && X < cols) {
							if (detectedBin.at<uchar>(Y, X) > 0) find = true;
						}
					}
				}
				if (find) true_positifs++;
				else faux_negatifs++;
			}
		}
	}

	cout << "\nMesures par contour:" << endl;
	cout << "contours_detectes = " << (true_positifs + faux_positifs) << endl;
	cout << "contours_reference = " << (true_positifs + faux_negatifs) << endl;
	cout << "contours_corrects = " << true_positifs << endl;
	cout << "faux_positifs = " << faux_positifs << endl;
	cout << "faux_negatifs = " << faux_negatifs << endl << endl;

	float denom = (true_positifs + faux_positifs + faux_negatifs);
	cout << "Mesures de performances:" << endl;
	if (denom > 0.0f) {
		cout << "Performance: " << P(faux_negatifs, true_positifs, faux_positifs) << endl;
		cout << "taux de faux positifs: " << TFP(faux_negatifs, true_positifs, faux_positifs) << endl;
		cout << "taux de faux negatifs: " << TFN(faux_negatifs, true_positifs, faux_positifs) << endl << endl;
	}
	else {
		cout << "Performance: - (aucune composante pour calculer)" << endl << endl;
	}
}

void runSobel(const cv::Mat& image, const cv::Mat& imageBin) {
	using namespace cv;
	using namespace std;

	cv::Mat gray = image.clone();
	if (image.channels() == 3) {
		cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
	}
	Mat sobelX;
	Mat sobelY;
	Sobel(gray, sobelX, CV_16S, 1, 0, 3);
	Sobel(gray, sobelY, CV_16S, 0, 1, 3);
	Mat sobel = abs(sobelX) + abs(sobelY);

	Sobel(gray, sobelX, CV_32F, 1, 0);
	Sobel(gray, sobelY, CV_32F, 0, 1);

	double sobmin, sobmax;
	minMaxLoc(sobel, &sobmin, &sobmax);

	Mat sobelImage;
	if (sobmax == 0) sobmax = 1.0;
	sobel.convertTo(sobelImage, CV_8U, -255.0 / sobmax, 255);

	Mat sobelThresholded;
	threshold(sobelImage, sobelThresholded, SOBEL_THRESH, 255, THRESH_BINARY);

	countContours(sobelThresholded, imageBin);
	namedWindow("Sobel Thresholded Image");
	imshow("Sobel Thresholded Image", sobelThresholded);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return;
};

class LaplacianZC {
private:
	cv::Mat laplace;
	int aperture;

public:
	LaplacianZC() :aperture(3) {}
	void setAperture(int a) { aperture = a; }
	int getAperture() const { return aperture; }
	cv::Mat computeLaplacian(const cv::Mat& image) {
		cv::Laplacian(image, laplace, CV_32F, aperture);
		return laplace;
	}
	cv::Mat getLaplacianImage(double scale = 1.0) {
		if (scale < 0) {
			double lapmin, lapmax;
			cv::minMaxLoc(laplace, &lapmin, &lapmax);
			scale = 127 / std::max(-lapmin, lapmax);
		}
		cv::Mat laplaceImage;
		laplace.convertTo(laplaceImage, CV_8U, scale, 128);
		return laplaceImage;
	}
	cv::Mat getZeroCrossings(cv::Mat laplace) {
		cv::Mat signImage;
		cv::threshold(laplace, signImage, 0, 255, cv::THRESH_BINARY);
		cv::Mat binary;
		signImage.convertTo(binary, CV_8U);
		cv::Mat dilated;
		cv::dilate(binary, dilated, cv::Mat());
		return dilated - binary;
	}
};

void runLaplacian(const cv::Mat& image, const cv::Mat& imageBin) {
	using namespace cv;
	using namespace std;
	Mat gaussMin, gaussMax, dog;
	Mat laplace;
	LaplacianZC laplacian;

	Mat gray = image.clone();
	if (image.channels() == 3) {
		cvtColor(gray, gray, COLOR_BGR2GRAY);
	}
	laplacian.setAperture(3);

	Mat lap = laplacian.computeLaplacian(gray);
	laplace = laplacian.getLaplacianImage();

	GaussianBlur(gray, gaussMin, Size(), LAPLACE_GAUSS_MIN);
	GaussianBlur(gray, gaussMax, Size(), LAPLACE_GAUSS_MAX);
	subtract(gaussMax, gaussMin, dog, Mat(), CV_32F);

	Mat zeros = laplacian.getZeroCrossings(dog);
	namedWindow("Laplacian Zero-crossings of DoG");
	imshow("Laplacian Zero-crossings of DoG", 255 - zeros);
	cv::waitKey(0);
	cv::destroyAllWindows();

	countContours(zeros, imageBin);
	cout << endl << endl;
}

void runCanny(const cv::Mat& image, const cv::Mat& imageBin) {
	using namespace cv;
	Mat gray;
	if (image.channels() == 3) cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	else gray = image.clone();

	Mat contours;
	Canny(gray, contours, CANNY_LOW_THRESH, CANNY_HIGH_THRESH);

	namedWindow("Canny Contours");
	imshow("Canny Contours", 255 - contours);
	waitKey(0);
	destroyAllWindows();

	countContours(contours, imageBin);
	return;
}

// Permettant de régler les paramètres associés
void fixParams(const cv::Mat& image) {
	using namespace std;
	using namespace cv;
	int option = -1;
	while (option != 0) {
		cout << "Choisissez un détecteur pour modifier les paramètres\n0: Retour\n1: Sobel\n2: Laplace\n3: Canny" << endl;
		cin >> option;
		switch (option) {
		case 1: {
			cout << "Detecteur Sobel choisi." << endl;
			cout << "SOBEL_THRESH (actuel = " << SOBEL_THRESH << ")" << endl;
			cout << "Saisissez SOBEL_THRESH (ou 0 pour conserver): ";
			int t; cin >> t;
			if (t > 0) SOBEL_THRESH = t;
			break;
		} case 2: {
			cout << "Detecteur Laplace choisi." << endl;
			cout << "LAPLACE_GAUSS_MIN (actuel = " << LAPLACE_GAUSS_MIN << "), LAPLACE_GAUSS_MAX (actuel = " << LAPLACE_GAUSS_MAX << ")" << endl;
			cout << "Saisissez LAPLACE_GAUSS_MIN LAPLACE_GAUSS_MAX (ou 0 0 pour conserver): ";
			float m, M; cin >> m >> M;
			if (m > 0) LAPLACE_GAUSS_MIN = m;
			if (M > 0) LAPLACE_GAUSS_MAX = M;
			break;
		} case 3: {
			cout << "Detecteur Canny choisi." << endl;
			cout << "CANNY_LOW_THRESH (actuel = " << CANNY_LOW_THRESH << "), CANNY_HIGH_THRESH (actuel = " << CANNY_HIGH_THRESH << ")" << endl;
			cout << "Saisissez CANNY_LOW CANNY_HIGH (ou 0 0 pour conserver): ";
			int l, h; cin >> l >> h;
			if (l > 0) CANNY_LOW_THRESH = l;
			if (h > 0) CANNY_HIGH_THRESH = h;
			break;
		} case 0: {
			cout << "Revenant..." << endl;
			break;
		}
		default: {
			cout << "Option invalide. Veuillez choisir entre 0,1,2 ou 3." << endl;
			break;
		}
		}
	}
}

// Capable de traiter « un lot d’images » 
void processingTons() {
	using namespace cv;
	using namespace std;
	string pathPaste = "IIA_images/images/contours/images/";
	vector<string> nameFiles = {
		"basket","bear","brush","buffalo","elephant","gazelle","gnu","goat",
		"golfcart","hyena","lions","rino","tiger","tire","turtle",
	};
	for (const auto& nameFile : nameFiles) {
		string fullPath = pathPaste + nameFile + ".pgm";
		string fullPathBin = pathPaste + "gt/" + nameFile + "_gt_binary.pgm";
		Mat image = imread(fullPath, IMREAD_GRAYSCALE);
		if (image.empty()) {
			cerr << "Erreur de chargement de l'image: " << fullPath << endl;
			continue;
		}
		Mat imageBin = imread(fullPathBin, IMREAD_GRAYSCALE);
		if (imageBin.empty()) {
			cerr << "Erreur de chargement de l'image: " << fullPathBin << endl;
			continue;
		}
		cout << "Processing image: " << nameFile << endl;
		namedWindow("Normal Image: " + nameFile);
		imshow("Normal Image: " + nameFile, image);
		cv::waitKey(0);
		cv::destroyAllWindows();

		runSobel(image, imageBin);
		runLaplacian(image, imageBin);
		runCanny(image, imageBin);
	}
	return;
}

int main() {
	using namespace std;
	using namespace cv;
	int option = 1;
	int detecteur = 0;
	Mat image, imageBin;
	string path = "IIA_images/images/contours/images/basket.pgm";
	string binPath = "IIA_images/images/contours/images/gt/basket_gt_binary.pgm";

	image = imread(path, IMREAD_GRAYSCALE);
	if (image.empty()) {
		cerr << "Erreur de chargement de l'image" << endl;
		return -1;
	}
	imageBin = imread(binPath, IMREAD_GRAYSCALE);
	if (imageBin.empty()) {
		cerr << "Erreur de chargement de l'image" << endl;
		return -1;
	}
	cout << "Image chargee avec succes" << endl;
	while (option != 0) {
		cout << "Choisissez ce que vous voudriez faire:\n0: Quitter \n1: Choisir le detecteur \n2: Regler les parametres associes \n3: Traiter un lot d'images\n4: Visualiser l'image original" << endl;
		cin >> option;
		switch (option) {
		case 0:
			cout << "Quitter le programme." << endl;
			break;
		case 1: {
			// Permettant de choisir le détecteur
			cout << "Choisir le detecteur." << endl;
			cout << "1: Sobel\n2: Laplace\n3: Canny" << endl;
			cin >> detecteur;
			switch (detecteur) {
			case 1: {
				cout << "Detecteur Sobel choisi." << endl;
				runSobel(image, imageBin);
				break;
			} case 2: {
				cout << "Detecteur Laplace choisi." << endl;
				runLaplacian(image, imageBin);
				break;
			} case 3: {
				cout << "Detecteur Canny choisi." << endl;
				runCanny(image, imageBin);
				break;
			} default: {
				cout << "Option invalide. Veuillez choisir entre 1, 2 ou 3." << endl;
				break;
			}
			}
			break;
		}
		case 2:
			cout << "Regler les parametres associes." << endl;
			fixParams(image);
			break;
		case 3:
			cout << "Traiter un lot d'images." << endl;
			processingTons();
			break;
		case 4:
			cout << "Visualiser l'image originale." << endl;
			namedWindow("Image");
			imshow("Image", image);
			waitKey(0);
			cv::destroyAllWindows();
			break;
		default:
			cout << "Option invalide. Veuillez reessayer." << endl;
			break;
		}
	}
	cout << "Fin du programme." << endl;
	return 0;
}
