

//Including to use ap_uint<> datatype
#include <ap_int.h>

#include "../../wide_vadd/src/definitions.hpp"

typedef ap_uint<512> uint512_dt;
#define VECTOR_SIZE (512 / 32) // VECTOR_SIZE size is 16 (512/32 = 16)

/*
    Matrix Multiplication Kernel Implementation
    Arguments:
        A   (input)     --> Input 	Matrix A
        B   (input)     --> Input 	Matrix B
        C   (output)    --> Output 	Matrix C = A*B
*/

extern "C" {
void MATRIX_MUL(const 	uint512_dt A[n * m / VECTOR_SIZE] , 	// Read-Only Matrix A
				const	uint512_dt B[m * p / VECTOR_SIZE] , 	// Read-Only Matrix B
						uint512_dt C[n * p / VECTOR_SIZE]		// Output Result Matrix C = A*B
					){

	// ========================= OURS ========================= //
	#pragma HLS INTERFACE m_axi port = A bundle = gmem
	#pragma HLS INTERFACE m_axi port = B bundle = gmem1
	#pragma HLS INTERFACE m_axi port = C bundle = gmem2
	#pragma HLS INTERFACE s_axilite port = A bundle = control
	#pragma HLS INTERFACE s_axilite port = B bundle = control
	#pragma HLS INTERFACE s_axilite port = C bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control

	uint512_dt ker_A [n*m / VECTOR_SIZE ];
	uint512_dt ker_B [m*p / VECTOR_SIZE ];
	// uint512_dt ker_C [n*p / VECTOR_SIZE ];


	copyA:
	for ( int i = 0 ; i < n*m/VECTOR_SIZE ; i++){
		#pragma HLS PIPELINE II = 1
		ker_A[i] = A[i];
	}
	copyB:
	for ( int i = 0 ; i < m*p/VECTOR_SIZE ; i++){
		#pragma HLS PIPELINE II = 1
		ker_B[i] = B[i];
	}

//	#pragma HLS array_partition type=block 		variable=ker_A 	factor= n
//	#pragma HLS array_partition type=cyclic 	variable=ker_B  factor= p
//	#pragma HLS array_partition type=cyclic 	variable=ker_C 	factor= p

	loop1:
	for( int i = 0 ; i < n ; i++ ){

		uint512_dt tmpOut = 0;

		loop2:
		for ( int k = 0 ; k < p ; k++ ){
			#pragma HLS PIPELINE II=1

			num_t_res temp = 0 ;


			loop3:
			for ( int j = 0 ; j < m ; j++ ){
				// We know m = 16 
				ap_int<32> tmp1 = ker_A[i*m].range( 32 * (j + 1) - 1,  j * 32);
				ap_int<32> tmp2 = ker_B[k*p].range( 32 * (j + 1) - 1,  j * 32);
				
				temp += tmp1 * tmp2;
				// tmpOut.range(32 * (k + 1) - 1, k * 32) += tmp1 * tmp2; 

				// temp += ker_A[i*m+j]*ker_B[k*p+j] ;	// with B = BT transpose

			}
			// C[i*p+k] = temp ; // num_t_res 
			tmpOut.range(32 * (k + 1) - 1, k * 32) = temp ; 
		}
	C[i] = tmpOut ; // uint512_dt
	}

//	loop1:
//	for( int i = 0 ; i < n ; i++ ){
//		loop2:
//		for ( int k = 0 ; k < p ; k++ ){
//			#pragma HLS PIPELINE II=1
//
//			num_t_res temp = 0 ;
//
//			loop3:
//			for ( int j = 0 ; j < m ; j++ ){
////				#pragma HLS unroll factor= m 	// This is auto added
//
//				temp += ker_A[i*m+j]*ker_B[j*p+k] ;
//			}
//			C[i] = temp ;
//		}
//	}

//	for ( int i =0 ; i < n*p ;i++){
//		#pragma HLS PIPELINE II = 1
////		Move this in the triple for loop above for performance
//		C[i] = ker_C[i];
//	}



} // Close function
} // Close C extern

