/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#define TILE_WIDTH 16


// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    __shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // Identify the row and column of the P_d element to work on
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;
  
    // Loop over the M_ and N_ tiles to compute the P_d element
    for (int m = 0; m < ceilf(M.width/(float)TILE_WIDTH); ++m) {
       
        // Collaborative loading of M_d and N_d tiles into shared memory
        if(m*blockDim.x + tx < M.width){
		    M_s[ty][tx] = M.elements[Row * M.width + m*TILE_WIDTH+tx];
        }
	    if(m*blockDim.y + ty < N.height) {
	    	N_s[ty][tx] = N.elements[(m*blockDim.y+ty)*N.width+Col];
        }
	__syncthreads();
        
        for (int k = 0; k < blockDim.y; ++k){
            if(m*blockDim.x+k < M.width){
                Pvalue += M_s[ty][k] * N_s[k][tx];
            }
        }

        __syncthreads();   
    }    
    if(Row < P.height && Col < P.width) {
        P.elements[Row*P.width+Col] = Pvalue;
    }

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
