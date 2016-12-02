#include <cstdio>
#include <iostream>

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#define dbg(x) cout <<#x<<"="<<x<<endl
#define dbg2(x,y) cout <<#x<<"="<<x<<" " <<#y<<"="<<y<<endl

using namespace std;
using namespace cv;


Mat NNI(Mat img, int rsz) {

	Mat out(img.rows * rsz,img.cols * rsz , CV_8UC3);

	//Vectorizando
	vector<Vec3b> v;
	
	for (int j = 0; j < img.rows; j++)
		for (int i = 0; i < img.cols; i++)
			v.push_back(img.at<Vec3b>(Point(i, j)));

	// Nearest Neighbor Interpolation
	int cont = 0;
	for (int j = 0; j < out.rows; j += rsz) {
		for (int i = 0; i < out.cols; i += rsz) {
			// Pintar vecinos i,j
			for (int dy = 0; dy < rsz;dy++) {
				for (int dx = 0; dx < rsz;dx++) {
					out.at<Vec3b>(Point(i+dx, j+dy)) = v[cont];
				}
			}
			cont++;
		}
	}

	return out;
}


Mat BilImgS(Mat img, int rsz) {

	Mat out(img.rows*rsz+1, img.cols*rsz+1, CV_8UC3);

	Vec3d A, B, C, D, offset, Y;
	float x_diff, y_diff;
	int dx = 0; int dy = 0;

	for (int j = rsz; j < out.rows; j += rsz) {
		dx = 0;
		for (int i = rsz; i < out.cols; i += rsz) {
			out.at<Vec3b>(Point(i , j)) = img.at<Vec3b>(Point(dx, dy));
			dx++;
		}
		dy++;
	}

	dx = 1, dy = 1;
	for (int j = rsz; j < out.rows-1; j += rsz) {
		dx = 1;
		for (int i = rsz; i < out.cols-1; i += rsz) {
			// Sacando los pixels
			//dbg2(dx, dy);
			A = img.at<Vec3b>(Point(dx, dy));
			B = img.at<Vec3b>(Point(dx-1, dy));
			C = img.at<Vec3b>(Point(dx , dy-1));
			D = img.at<Vec3b>(Point(dx-1, dy-1));

			// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
			
			for (int dyy = j-rsz; dyy <= j;dyy++) {
				for (int dxx = i-rsz; dxx <= i; dxx++) {
					//dbg2(dxx,dyy);
					float w = 1-((j - dyy) / (rsz+1));
					float h = 1-((i - dxx) / (rsz+1));

					// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
					Y[0] = (int)(A[0] *(1 - w)*(1 - h) + B[0] *(w)*(1 - h) + C[0] *(h)*(1 - w) + D[0] *(w*h));
					Y[1] = (int)(A[1] * (1 - w)*(1 - h) + B[1] * (w)*(1 - h) + C[1] * (h)*(1 - w) + D[1] * (w*h));
					Y[2] = (int)(A[2] * (1 - w)*(1 - h) + B[2] * (w)*(1 - h) + C[2] * (h)*(1 - w) + D[2] * (w*h));

					out.at<Vec3b>(Point(dxx, dyy)) = Y;

				}
			}

			dx++;
		}
		dy++;
	}


	return out;
}

int main() {
	Mat img = imread("Sample.jpg", CV_LOAD_IMAGE_COLOR);
	if (!img.data) { cout << "No se puede abrir o encontrar esta imagen\n" << std::endl;
		return -1;
	}

	int w1 = img.rows;
	int h1 = img.cols;

	int rsz = 3; // veces que crecera la imagen
	//img = NNI(img,rsz);

	img = BilImgS(img,rsz);

	imshow("Montando",img);

	imwrite("Salida_google.jpg",img);

	waitKey(0);
	return 0;
}