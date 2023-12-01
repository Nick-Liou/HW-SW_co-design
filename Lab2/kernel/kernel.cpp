/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce
latency and
    device resource utilization of the resulting RTL code
    This is vector addition example to demonstrate how HLS optimizations are
used in kernel.
*******************************************************************************/


#include "../../Matrix_Multiplication/src/definitions.hpp"


// This will do FULL unrolling EVERYTHING
#define SUPER_MEGA_ULTRA_OPTIMIZATIONS false && OPTIMIZATIONS_CONDITION
//
//#define BUFFER_SIZE 1024
//#define DATA_SIZE 4096
//
//// TRIPCOUNT identifier
//const unsigned int c_len = DATA_SIZE / BUFFER_SIZE;
//const unsigned int c_size = BUFFER_SIZE;

///*
//    Vector Addition Kernel Implementation
//    Arguments:
//        in1   (input)     --> Input Vector1
//        in2   (input)     --> Input Vector2
//        out_r   (output)    --> Output Vector
//        size  (input)     --> Size of Vector in Integer
//*/

	// Here Vitis kernel contains one s_axilite interface which will be used by host
	// application to configure the kernel.
	// Here bundle control is defined which is s_axilite interface and associated
	// with all the arguments (in1, in2, out_r and size),
	// control interface must also be associated with "return".
	// All the global memory access arguments must be associated to one m_axi(AXI
	// Master Interface). Here all three arguments(in1, in2, out_r) are
	// associated to bundle gmem which means that a AXI master interface named
	// "gmem" will be created in Kernel and all these variables will be
	// accessing global memory through this interface.
	// Multiple interfaces can also be created based on the requirements. For
	// example when multiple memory accessing arguments need access to
	// global memory simultaneously, user can create multiple master interfaces and
	// can connect to different arguments.
	//#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem
	//#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem
	//#pragma HLS INTERFACE m_axi port = out_r offset = slave bundle = gmem
	//#pragma HLS INTERFACE s_axilite port = in1 bundle = control
	//#pragma HLS INTERFACE s_axilite port = in2 bundle = control
	//#pragma HLS INTERFACE s_axilite port = out_r bundle = control
	//#pragma HLS INTERFACE s_axilite port = size bundle = control
	//#pragma HLS INTERFACE s_axilite port = return bundle = control

/*
    Matrix Multiplication Kernel Implementation
    Arguments:
        A   (input)     --> Input 	Matrix A
        B   (input)     --> Input 	Matrix B
        C   (output)    --> Output 	Matrix C = A*B
*/

extern "C" {
void MATRIX_MUL(const 	num_t  		A[n * m] , 	// Read-Only Matrix A
				const	num_t  		B[m * p] , 	// Read-Only Matrix B
						num_t_res  	C[n * p]	// Output Result Matrix C = A*B
					){

	// ========================= OURS ========================= //
	#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem
	#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem
	#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem
	#pragma HLS INTERFACE s_axilite port = A bundle = control
	#pragma HLS INTERFACE s_axilite port = B bundle = control
	#pragma HLS INTERFACE s_axilite port = C bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control

	num_t ker_A [n*m];
	num_t ker_B [m*p];
	num_t_res ker_C [n*p];

	for ( int i =0 ; i < n*m ;i++){
		#pragma HLS PIPELINE II = 1
		ker_A[i] = A[i];
	}
	for ( int i =0 ; i < m*p ;i++){
		#pragma HLS PIPELINE II = 1
		ker_B[i] = B[i];
	}

	#pragma HLS array_partition type=block 		variable=ker_A 	factor= n
	#pragma HLS array_partition type=cyclic 	variable=ker_B  factor= p
	#pragma HLS array_partition type=cyclic 	variable=ker_C 	factor= p

	loop1:
	for( int i = 0 ; i < n ; i++ ){
		loop2:
		for ( int k = 0 ; k < p ; k++ ){
			#pragma HLS PIPELINE II=1
//			#pragma HLS unroll factor= 4096/m

			num_t_res temp = 0 ;

			loop3:
			for ( int j = 0 ; j < m ; j++ ){
//				#pragma HLS unroll factor= m

				temp += ker_A[i*m+j]*ker_B[j*p+k] ;
			}
			ker_C[i*p+k] = temp ;
		}
	}

	for ( int i =0 ; i < n*p ;i++){
		#pragma HLS PIPELINE II = 1
//		Move this in the triple for loop above for performance
		C[i] = ker_C[i];
	}
	// ========================= VADD EXAMPLE =========================

//	  unsigned int v1_buffer[BUFFER_SIZE];   // Local memory to store vector1
//	  unsigned int v2_buffer[BUFFER_SIZE];   // Local memory to store vector2
//	  unsigned int vout_buffer[BUFFER_SIZE]; // Local Memory to store result
//
//	  // Per iteration of this loop perform BUFFER_SIZE vector addition
//		  for (int i = 0; i < size; i += BUFFER_SIZE) {
//		#pragma HLS LOOP_TRIPCOUNT min = c_len max = c_len
//			int chunk_size = BUFFER_SIZE;
//			// boundary checks
//			if ((i + BUFFER_SIZE) > size)
//			  chunk_size = size - i;
//
//		  // Transferring data in bursts hides the memory access latency as well as
//		  // improves bandwidth utilization and efficiency of the memory controller.
//		  // It is recommended to infer burst transfers from successive requests of data
//		  // from consecutive address locations.
//		  // A local memory vl_local is used for buffering the data from a single burst.
//		  // The entire input vector is read in multiple bursts.
//		  // The choice of LOCAL_MEM_SIZE depends on the specific applications and
//		  // available on-chip memory on target FPGA.
//		  // burst read of v1 and v2 vector from global memory
//
//		  read1:
//			for (int j = 0; j < chunk_size; j++) {
//		#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
//		#pragma HLS PIPELINE II = 1
//			  v1_buffer[j] = in1[i + j];
//			}
//
//		  read2:
//			for (int j = 0; j < chunk_size; j++) {
//		#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
//		#pragma HLS PIPELINE II = 1
//			  v2_buffer[j] = in2[i + j];
//			}
	//
	//	  // PIPELINE pragma reduces the initiation interval for loop by allowing the
	//	  // concurrent executions of operations
	//	  vadd:
	//		for (int j = 0; j < chunk_size; j++) {
	//	#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
	//	#pragma HLS PIPELINE II = 1
	//		  // perform vector addition
	//		  vout_buffer[j] = v1_buffer[j] + v2_buffer[j];
	//		}
	//
	//	  // burst write the result
	//	  write:
	//		for (int j = 0; j < chunk_size; j++) {
	//	#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
	//	#pragma HLS PIPELINE II = 1
	//		  out_r[i + j] = vout_buffer[j];
	//		}
	//	  }
	//	}


} // Close function
} // Close C extern

