#include <stdio.h>
#include <cuda.h>

using namespace std;

__device__ int volatile idx=-1;
__device__ unsigned volatile _time=0;

/*
The GPU algorithm is basically the same as the CPU algorithm, except that it uses a lock to force the threads to execute in order.
I found the GPU code to be about a 100 times slower than the CPU code.
*/


__global__ void dkernel(int * times, int * prior, int * core_time, int * core_prio, int m, int n){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i<n){
        while(i!=idx){
            if(idx==i-1){
                if(core_prio[prior[i]]==-1){
                    int min_core = 0;
                    for(int j=0;j<m;j++){
                        min_core = core_time[j]<core_time[min_core]?j:min_core;
                        if(_time<core_time[j])continue;
                        core_time[j]=_time;
                        core_prio[prior[i]] = j;
                        break;
                    }
                    if(core_prio[prior[i]]==-1){
                        core_prio[prior[i]]=min_core;
                    }
                }
                int curr_core = core_prio[prior[i]];
                if(core_time[curr_core]>_time)_time=core_time[curr_core];
                else core_time[curr_core]=_time;
                core_time[curr_core]+=times[i];
                times[i] = core_time[curr_core];
                __threadfence();
                idx = i;
            }
        }
    }
}


//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    int *core_time, *h_core_time;
    int *core_prio, *h_core_prio;
    int *prior;
    int *times;
    
    h_core_time = (int*)malloc(m*sizeof(int));
    h_core_prio = (int*)malloc(m*sizeof(int));
    cudaMalloc(&core_time, m*sizeof(int));
    cudaMalloc(&core_prio, m*sizeof(int));
    cudaMalloc(&prior, n*sizeof(int));
    cudaMalloc(&times, n*sizeof(int));

    for(int i=0;i<m;i++)h_core_time[i]=0;
    for(int i=0;i<m;i++)h_core_prio[i]=-1;
    
    cudaMemcpy(core_time, h_core_time, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(core_prio, h_core_prio, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(prior, priority, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(times, executionTime, n*sizeof(int), cudaMemcpyHostToDevice);

    dkernel<<<((1023+n)/1024), 1024>>>(times, prior, core_time, core_prio, m, n);
    
    cudaMemcpy(result , times, n*sizeof(int), cudaMemcpyDeviceToHost);
}

void cpu_operations (int m, int n, int *times, int *prior, int * result){
    int core_time[m];
    int core_prio[m];
    for(int i=0;i<m;i++)core_time[i] = 0;
    for(int i=0;i<m;i++)core_prio[i] = -1;
    int time = 0;
    for(int i=0;i<n;i++){
        if(core_prio[prior[i]]==-1){
            int min_core = 0;
            for(int j=0;j<m;j++){
                min_core = core_time[j]<core_time[min_core]?j:min_core;
                if(time < core_time[j])continue;
                core_time[j] = time;
                core_prio[prior[i]] = j;
                break;
            }
            if(core_prio[prior[i]]==-1){
                core_prio[prior[i]] = min_core;
            }
        }
        int curr_core = core_prio[prior[i]];
        if(core_time[curr_core] > time)time = core_time[curr_core];
        else core_time[curr_core] = time;
        core_time[curr_core] += times[i];
        result[i] = core_time[curr_core];
//        printf("%d ",core_time[curr_core]);
    }
//    int prio_core[N];
//    for(int i=0;i<N;i++)prio_core[i]=0;
//    for(int i=0;i<M;i++){
//        prio_core[core_prio[i]]+=1;
//        if(prio_core[core_prio[i]]==2)printf("Sad life");
//    }
    
}


int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }
    printf("Taking inputs\n");
    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }
    
   
    printf("Inputs taken;");
   //Taking execution time and priorities as input	
    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );

    free(executionTime);
    free(priority);
    free(result);
    
    
    
}
