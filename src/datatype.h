/* 
 * This file is developed by Xuanzhi LIU (Walker LAU).
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

#ifndef DATATYPE_H
#define DATATYPE_H

#include "sdsoc.h"

#ifdef SDSOC
    //#include "ap_fixed.h"
    #include "hls_half.h"
	typedef half  data_1;	//存输入体、输出体	    //typedef float data_1; 	//typedef ap_fixed<8,  6, AP_TRN_ZERO, AP_SAT>  data_1;
	typedef half  data_2; 	//存卷积核			   //typedef float data_2;		//typedef ap_fixed<8,  0, AP_TRN_ZERO, AP_SAT>  data_2;
	typedef half  data_3;	//存乘积、累加和	    //typedef float data_3;		//typedef ap_fixed<8,  3, AP_TRN_ZERO, AP_SAT>  data_3;
#else
    //#include "half.hpp"
    //using half_float::half;
    typedef float data_1;     //typedef half data_1;   //typedef make_fixed<2, 5>    data_1;     //typedef float data_1;     //typedef int8_t  data_1;
    typedef float data_2;     //typedef half data_2;   //typedef make_fixed<5, 10>   data_2;     //typedef float data_2;     //typedef int16_t data_2;
    typedef float data_3;
#endif  // SDSOC

#endif  // DATATYPE_H
