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


#include "../../wide_vadd/src/definitions.hpp"


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

			num_t_res temp = 0 ;

			loop3:
			for ( int j = 0 ; j < m ; j++ ){
//				#pragma HLS unroll factor= m 	// This is auto added

				temp += ker_A[i*m+j]*ker_B[j*p+k] ;
			}
			C[i] = temp ;
		}
	}

//	for ( int i =0 ; i < n*p ;i++){
//		#pragma HLS PIPELINE II = 1
////		Move this in the triple for loop above for performance
//		C[i] = ker_C[i];
//	}



} // Close function
} // Close C extern

