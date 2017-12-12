/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMul(__global float* C, 
          __global float* A, 
          __global float* B, 
          int wA, int wB)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[ty * wA + k];
      float elementB = B[k * wB + tx];
      value += elementA * elementB;
   }
   C[ty * wA + tx] = value;
}

__kernel void
msub(__global float* C,
  	 __global float* A,
	 float B,
	 int wA, int wB)
{
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   float value = B * A[ty * wA + tx];
   C[ty * wA + tx] -= value;

}

__kernel void
madd(__global float* C,
  	 __global float* A,
	 __global float* B,
	 __global float* D,
	 int wA, int wB)
{
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[ty * wA + k];
      float elementB = B[k * wB + tx];
      value += elementA * elementB;
   }
   C[tx * wB + ty] = value + D[tx * wB + ty];
}

__kernel void
vecvecmat_mult(__global float* C,
			   __global float* A,
			   __global float* B,
			   int wA, int wB)
{
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   C[ty * wA + tx] = A[ty] * B[tx];
}


__kernel void
trans_multProduct(__global float* C,
  	 __global float* A,
	 __global float* B,
	 __global float* D,
	 int wA, int wB)
{
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[tx * wB + k];
      float elementB = B[k * wB + tx] * D[k * wB + tx];
      value += elementA * elementB;
   }
   C[ty * wA + tx] = value;
}