/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Identification module, containing codes implementing the
 * face identification method described in the following paper:
 *
 *
 *   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
 *   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
 *   In Frontiers of Computer Science.
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Zining Xu(a M.S. supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 * -----------------------------------------------------------------------------------------------------
 * 
 * The FPGA acceleration parts of this file are developed by Xuanzhi LIU (Walker LAU).
 * 
 * This version accelerates all 7 CONV-layers of VIPLFaceNet.
 * 
 * If you want to get the latest version of this project or met any problems,
 * please go to <https://github.com/WalkerLau/Accelerating-CNN-with-FPGA> , 
 * I will try to help as much as I can.
 * 
 * You can redistribute this source codes and/or modify it under the terms of the BSD 2-Clause License.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 * 
 */

#ifndef MATH_FUNCTIONS_H_
#define MATH_FUNCTIONS_H_


#include "sdsoc.h"
#include "log.h"
#include "datatype.h"

#ifdef _BLAS
#ifdef _WIN64
#pragma comment( lib, "blas_win64_MT" )
#pragma comment( lib, "lapack_win64_MT" )
#else
#pragma comment( lib, "libblas" )
#pragma comment( lib, "liblapack" )
#endif
#include <armadillo>
#endif

float simd_dot(const float* x, const float* y, const long& len);

#ifdef SDSOC
#pragma SDS data mem_attribute(A:PHYSICAL_CONTIGUOUS, B:PHYSICAL_CONTIGUOUS, C:PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(A:AXIDMA_SIMPLE, B:AXIDMA_SIMPLE, C:AXIDMA_SIMPLE)
#pragma SDS data access_pattern(A:SEQUENTIAL, B:SEQUENTIAL, C:SEQUENTIAL )
#endif
void matrix_procuct(const data_1 A[128*29*29], const data_2 B[192*256*3*3], data_1 C[128*27*27], const int m, const int k); 

#endif // MATH_FUNCTIONS_H_
