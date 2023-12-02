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

#include "xcl2.hpp"
#include "event_timer.hpp"
#include <algorithm>
#include <vector>

//#define DATA_SIZE 4096


#include <stdio.h>
#include <cstdlib>


#include "definitions.hpp"  // This should be included last!!! (to avoid problems with the Defines)

#define PRINT_ALL_MATRICES



void fill_with_rand( num_t * A , int dim1 , int dim2  );

template<typename T>
void print_matrix( T * A , int dim1 , int dim2 );

//void MATRIX_MUL( num_t * A , num_t * B , num_t_res * C );

template<typename T1 , typename T2 , typename T3>
void matrix_mul_soft( T1 * A , T2 * B , T3 * C , int dim1 , int dim2, int dim3 );

template<typename T>
bool test_matrix_equality( T* A , T* B , int dim1 , int dim2 ) ;

template<typename T>
void transpose_matrix( T * A , T* AT , int dim1 , int dim2 );

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  EventTimer et;

  std::string binaryFile = argv[1];
//  size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
  cl_int err;
  cl::Context context;
  cl::Kernel krnl_matrix_mult;
  cl::CommandQueue q;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
  // hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice
  // but to create
  // its own host side buffer. So it is recommended to use this allocator if
  // user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
  // boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with
  // CL_MEM_USE_HOST_PTR

  // ===================== Initialize ===================== //
  et.add("Allocate Memory in Host Memory");
  std::vector<num_t, 	aligned_allocator<num_t>> 		A(n * m);
  std::vector<num_t, 	aligned_allocator<num_t>> 		B(m * p);
  std::vector<num_t, 	aligned_allocator<num_t>> 		BT(m * p);
  std::vector<num_t_res,aligned_allocator<num_t_res>> 	C_SW(n * p);
  std::vector<num_t_res,aligned_allocator<num_t_res>> 	C_HW(n * p);
  et.finish();

  printf("n = %d \n" ,n);
  printf("m = %d \n" ,m);
  printf("p = %d \n" ,p);


  // Create the test data
  et.add("Fill the buffers");


  fill_with_rand(A.data() , n , m ) ;
  fill_with_rand(B.data() , m , p ) ;
  
  transpose_matrix(B.data(), BT.data() , m , p) ;

  // ===================== Prints ===================== //

	#ifdef PRINT_ALL_MATRICES

	  printf("Matrix A = \n");
	  print_matrix(A.data() , n , m ) ;

	  printf("Matrix B = \n");
	  print_matrix(B.data() , m , p ) ;
	#endif


  // Multiplication on S/W
  matrix_mul_soft(A.data(),B.data(),C_SW.data(),n,m,p) ;

	#ifdef PRINT_ALL_MATRICES
	  printf("Matrix (software) C = \n");
	  print_matrix(C_SW.data() , n , p ) ;
	#endif


  et.finish();

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  et.add("Load Binary File to Alveo U200");
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  int valid_device = 0;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err, krnl_matrix_mult = cl::Kernel(program, "MATRIX_MUL", &err));
      valid_device++;
      break; // we break because we found a valid device
    }
  }
  if (valid_device == 0) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }
  et.finish();

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  et.add("Allocate Buffer in Global Memory");
  OCL_CHECK(err, cl::Buffer buffer_A(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
					 A.size()*sizeof(num_t), A.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_B(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
					 BT.size()*sizeof(num_t), BT.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_C(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
					 C_HW.size()*sizeof(num_t_res), C_HW.data(), &err));
  et.finish();

//  int dummy_input = 0 ;
  et.add("Set the Kernel Arguments");
  OCL_CHECK(err, err = krnl_matrix_mult.setArg(0, buffer_A));
  OCL_CHECK(err, err = krnl_matrix_mult.setArg(1, buffer_B));
  OCL_CHECK(err, err = krnl_matrix_mult.setArg(2, buffer_C));
//  OCL_CHECK(err, err = krnl_matrix_mult.setArg(3, dummy_input));
  et.finish();



  // Copy input data to device global memory
  et.add("Copy input data to device global memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A, buffer_B}, 0 /* 0 means from host*/));
  et.finish();

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel
  et.add("Launch the Kernel");
  OCL_CHECK(err, err = q.enqueueTask(krnl_matrix_mult));
  et.finish();

  // Copy Result from Device Global Memory to Host Local Memory
  et.add("Copy Result from Device Global Memory to Host Local Memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_C}, CL_MIGRATE_MEM_OBJECT_HOST));
  OCL_CHECK(err, err = q.finish());
  et.finish();
  // OPENCL HOST CODE AREA END


	#ifdef PRINT_ALL_MATRICES
		  printf("Matrix (hardware) C = \n");
		  print_matrix(C_HW.data() , n , p ) ;
	#endif

  // Compare the results of the Device to the simulation
  et.add("Compare the results of the Device to the simulation");
  bool test_passed = test_matrix_equality( C_SW.data() , C_HW.data(), n , p ) ;
  et.finish();

  std::cout <<"----------------- Key execution times -----------------" << std::endl;
  et.print();

  std::cout << "TEST " << (test_passed ? "PASSED" : "FAILED") << std::endl;
  return (test_passed ? EXIT_SUCCESS : EXIT_FAILURE);
}





void fill_with_rand( num_t * A , int dim1 , int dim2 ){

    for ( int i = 0 ; i < dim1 * dim2; i++ ){
        A[i] = (num_t) rand()%255;
    }

}


template<typename T>
void print_matrix( T * A , int dim1 , int dim2 ){

    for ( int i = 0 ; i < dim1; i++ ){
        printf( "[" ) ;
        for ( int j = 0 ; j < dim2; j++ ){
            printf( " %7d " , A[i*dim2+j]  ) ;
        }
        printf( "]\n" ) ;
    }

    printf( "\n" ) ;

}


template<typename T>
void transpose_matrix( T * A , T* AT , int dim1 , int dim2 ){

    for ( int i = 0 ; i < dim1; i++ ){
        for ( int j = 0 ; j < dim2; j++ ){
            AT [j*dim1+i ] = A[i*dim2+j] ; 
        }
    }

}





template<typename T1 ,typename T2 ,typename T3>
void matrix_mul_soft( T1 * A , T2 * B , T3 * C , int dim1 , int dim2, int dim3 ){

    for( int i = 0 ; i < dim1 ; i++ ){
            for ( int k = 0 ; k < dim3 ; k++ ){
        	T3 sum = 0 ;
        for ( int j = 0 ; j < dim2 ; j++ ){
				sum += A[i*dim2+j]*B[j*dim3+k] ;
            }
			C[i*dim3+k] = sum ;
        }
    }

}

template<typename T>
bool test_matrix_equality( T* A , T* B , int dim1 , int dim2 ){

    bool flag = true ;
    for ( int i = 0 ; i < dim1 * dim2 ; i++) {
        if ( A[i] != B[i] ){
            flag =  false ;
            printf("Not equal elements! A[%d][%d] = %7d , B[%d][%d] = %7d \n" , i/dim1 , i%dim1 , A[i] , i/dim1 , i%dim1 , B[i]  ) ;
        }
    }

    if ( flag ){
        printf("Test Passed\n");
    }
    else{
        printf("Test Failed\n");
    }
    return flag;
}






