#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "mypgm.h"

#define SIZE 4096
#define MAXN 4096*4096
#define N 16384
#define L 1024
#define INF (1<<30)
#define MIN_(a,b) (((a)<(b))?(a):(b))
#define MAX_(a,b) (((a)>(b))?(a):(b))
#define sqr(x) (x)*(x)

typedef float uc;
float minimo = INF;
float maximo = -INF;

__global__ void kernel(uc *img1, uc *img2, float *mask, int *radio){
	int offset = (blockIdx.x * L) + threadIdx.x;
	float p, m;
	float suma = 0;
	int radio2;
	
	p = 0.0;
	int idx = offset, cont = 0;

	radio2 = *radio / 2;
	suma = 0;
	int q = 0;

	for (int h = -radio2; h <= radio2; h++)
	{
		for (int r = 0; r < *radio; r++)
		{
			p = img1[idx + SIZE*h + r - 1];
			m = mask[q]; 
			q++;
			suma += p*m;
		}
	}

	img2[idx] = suma;
}

uc* vectorize(){
	uc *img = new uc[SIZE*SIZE];
	for (size_t i = 0; i < SIZE; i++)
		for (size_t j = 0; j < SIZE; j++){
			size_t pos = (i*SIZE) + j;
			img[pos] = image1[i][j];
		}
	return img;
}

int* values(int r){
	int *img = new int[r*r];
	for (size_t i = 0; i < r; i++)
		for (size_t j = 0; j < r; j++){
			size_t pos = (i*r) + j;
			img[pos] = 1;
		}
	return img;
}

void vector_to_matrix(uc* img){
	for (size_t i = 0; i < SIZE; i++)
		for (size_t j = 0; j < SIZE; j++){
			size_t pos = SIZE*i + j;
			image2[i][j] = img[pos];
		}
}


int main(void)
{
	load_image_data();
	uc *dev_img1;
	uc *dev_img2;
	float *dev_mask;

	int r = 21, *dev_r;
	uc *img1 = vectorize();
	uc *img2 = new uc[SIZE*SIZE];
	float *mask = new float[r*r];

	for (auto i = 0; i < r*r; i++)
		mask[i] = (float)1.0 / (float)(r*r);



	cudaMalloc((void**)&dev_img1, SIZE*SIZE*sizeof(float));
	cudaMalloc((void**)&dev_img2, SIZE*SIZE*sizeof(float));
	cudaMalloc((void**)&dev_mask, r*r*sizeof(float));
	cudaMalloc((void**)&dev_r, sizeof(int));

	cudaMemcpy(dev_img1, img1, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mask, mask, r * r * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, &r, sizeof(int), cudaMemcpyHostToDevice);

	float ms;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Haciendo el calculo de tiempo en el device
	kernel << <N, L >> >(dev_img1, dev_img2, dev_mask, dev_r);
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("GPUassert: %s\n", cudaGetErrorString(cudaGetLastError()));


	double memXFers = 5 * 4 * 4096 * 4096;
	memXFers /= 1024 * 1024 * 1024;
	printf("GPU: %f milisegundos con un ancho de banda %g GB/s\n", ms, memXFers / (ms / 1000.0));
	printf("Segundos: %f\n",ms/1000);

	cudaMemcpy(img2, dev_img2, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);

	vector_to_matrix(img2);

	x_size2 = SIZE;
	y_size2 = SIZE;

	save_image_data();

	return 0;
}