#include <cstdio>
#include <iostream>
#include <time.h>
#include <vector>
#include <omp.h>
#define dbg(x) std::cout << #x << "=" << x << std::endl
#include "mypgm.h"

using namespace std;

#define sqr(x) (x)*(x)


int conv2dPar( const vector<vector<int>> &kernel, const int &dim, const int &r, const int &suma){
	float sum;
	clock_t t;

	t = clock();

#pragma omp parallel for private(sum)
	for (int i = r; i<dim - r; i++){
		for (int j = r; j<dim - r; j++){
			sum = 0;
			for (int m = -r; m <= r; m++){
				for (int n = -r; n <= r; n++){
					sum += (int)image1[i + m][j + n] * kernel[m + r][n + r]/suma;
				}
			}

			image2[i][j] = (unsigned char)sum;

			//dbg(image2[i][j]);

		}
	}
	t = clock() - t;
	printf("Tiempo convolucion 2D Paralelo: %f\n", (float)t / CLOCKS_PER_SEC);

	return 0;
}

void conv2d(){

	int dim = 512;
	int r = 4;

	vector<vector<int>> kernel(2 * r + 1);
	for (int i = 0; i < 2 * r + 1; i++){
		kernel[i].resize(2 * r + 1);
	}

	int suma = 0;
	for (int i = 0; i < 2 * r + 1; i++){
		for (int j = 0; j < 2 * r + 1; j++){
			kernel[i][j] = 1;
			suma += kernel[i][j];
		}
	
	}
	//conv2dSerial(img, kernel, dim, r);
	//dbg(true);
	conv2dPar(kernel, dim, r,suma);

}

int main(){

	load_image_data();   /* Leer entrada en image1*/

	dbg(true);
	conv2d(); 

	save_image_data();   /* Salida de image2 */

	return 0;
}