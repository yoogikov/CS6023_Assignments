#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


// write kernels here...
__global__ void addT(int *A, int *B, int p, int q){
    unsigned id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id>=p*q)return;
    __shared__ unsigned x;
    x = id/q;
    __shared__ unsigned y;
    y = id%q;
    A[x*q+y] += B[y*p+x];
}

    
__global__ void mult(int * a, int * b, int * c, int p, int q, int r){
    int j = threadIdx.x;
    int i = blockIdx.x;
    __shared__ int arr[1024];
    if(j<q){
        arr[j] = a[i*q+j];
    }
    if(i>=p or j>=r)return;
    __syncthreads();
    int sum=0;
    for(int k=0;k<q;k++){
        sum += arr[k] * b[k*r+j];
    }
    c[i*r+j] = sum;
}

__global__ void transpose(int * b, int * bt, int p, int q){
    int i = blockIdx.x;
    int j = threadIdx.x;
    bt[j*p+i] = b[i*q+j];
}
// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations...
    int *A, *B, *C, *D, *Dt, *X, *Y;
    	
	// allocate memory...
    cudaMalloc(&A, p*q*sizeof(int));
    cudaMalloc(&B, p*q*sizeof(int));
    cudaMalloc(&C, q*r*sizeof(int));
    cudaMalloc(&D, s*r*sizeof(int));
    cudaMalloc(&Dt,s*r*sizeof(int));
    cudaMalloc(&X, p*s*sizeof(int));
	cudaMalloc(&Y, p*r*sizeof(int));

	// copy the values...
    cudaMemcpy(A, h_matrixA, p*q*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_matrixB, p*q*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(C, h_matrixC, r*q*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D, h_matrixD, r*s*sizeof(int), cudaMemcpyHostToDevice);
	
	// call the kernels for doing required computations...
    // Due to memory coalescing, I found that transposing and multiplying is faster when done in two steps.
    addT<<<ceil((p*q)/1024.0), 1024>>>(A, B, p, q);
    mult<<<p, max(q,r)>>>(A, C, Y, p, q, r);
    transpose<<<s, r>>>(D, Dt, s, r);
    mult<<<p, max(s, r)>>>(Y, Dt, X, p, r, s);

	// copy the result back...
    cudaMemcpy(h_matrixX, X, p*s*sizeof(int), cudaMemcpyDeviceToHost);
	
	// deallocate the memory...
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    cudaFree(X);
    cudaFree(Y);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int));

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
    gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}
