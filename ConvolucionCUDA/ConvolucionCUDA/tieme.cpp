#include "stdio.h"
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define COLUMNS 500
#define ROWS 500
long a[ROWS*COLUMNS], b[ROWS*COLUMNS], c[ROWS*COLUMNS],d[ROWS*COLUMNS];



__global__ void add(long *a, long *b, long *c,long *d)
{
 int x = blockIdx.x;
 int y = blockIdx.y;
 int i = (COLUMNS*y) + x;
 c[i] = a[i] + b[i];
 a[i]=d[i];
}
int main()
{
 long *dev_a, *dev_b, *dev_c,*dev_d;
 struct timeval startc, end;
 float ms;
 long seconds, useconds;
 double mtime;
 cudaEvent_t start,stop;


 for(int i=0; i<ROWS*COLUMNS; i++)
     d[i]=rand()%4;

 for(int i=0; i<ROWS; i++){
     for(int j=0; j<COLUMNS; j++){
         a[i*COLUMNS+j]=j;
         b[i*COLUMNS+j]=i;
     }
 }

 cudaMalloc((void **) &dev_a, ROWS*COLUMNS*sizeof(int));
 cudaMalloc((void **) &dev_b, ROWS*COLUMNS*sizeof(int));
 cudaMalloc((void **) &dev_c, ROWS*COLUMNS*sizeof(int));
 cudaMalloc((void **) &dev_d, ROWS*COLUMNS*sizeof(int));



 gettimeofday(&startc, NULL);
 for (long i = 0; i < ROWS*COLUMNS; i++){ // Fill Arrays
     c[i]=a[i]+b[i];
     a[i]=d[i];
 }
  gettimeofday(&end, NULL);

 seconds  = end.tv_sec  - startc.tv_sec;
 useconds = end.tv_usec - startc.tv_usec;
 mtime = useconds;
 mtime/=1000;
 mtime+=seconds*1000;

for (long y = ROWS-1; y < ROWS; y++) // Output Arrays
 {
 for (long x = COLUMNS-1; x < COLUMNS; x++)
 {
    // printf("\n[%ld][%ld]=%ld ",y,x,c[y][x]);
   //   printf("[%d][%d]=%d ",y,x,d[y][x]);
 }
 printf("\n");
 }



cudaMemcpy(dev_a, a, ROWS*COLUMNS*sizeof(int),
cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, ROWS*COLUMNS*sizeof(int),
cudaMemcpyHostToDevice);
cudaMemcpy(dev_d, d, ROWS*COLUMNS*sizeof(int),
cudaMemcpyHostToDevice);


cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);



add<<<dim3(1024,1024),dim3(128,128)>>>(dev_a, dev_b, dev_c,dev_d);



cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);

cudaMemcpy(c, dev_c, ROWS*COLUMNS*sizeof(int),cudaMemcpyDeviceToHost);
cudaEventDestroy(start);
cudaEventDestroy(stop);

printf("GPUassert: %s\n", cudaGetErrorString(cudaGetLastError()));

//cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
double memXFers=5*4*COLUMNS*ROWS;
memXFers/=1024*1024*1024;


printf("GPU: %f ms bandwidth %g GB/s",ms, memXFers/(ms/1000.0));
printf("\n CPU : %g ms bandwidth %g GB/s",mtime, memXFers/(mtime/1000.0));
 for (long y = ROWS-1; y < ROWS; y++) // Output Arrays
 {
     for (long x = COLUMNS-1; x < COLUMNS; x++)
     {
      //   printf("\n[%ld][%ld]=%ld ",y,x,c[y][x]);
      //   printf("[%d][%d]=%d ",y,x,d[y][x]);
     }
     printf("\n");
 }