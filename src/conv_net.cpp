/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
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

#include "conv_net.h"
#include "time.h"
#include "ctime"
#include "sdsoc.h"
#include "datatype.h"
#include "math_functions.h"
#include "convolute1.h"

#define CONV1 3*9*9*48	    // number of elements in filters of CONV1 is 3*9*9*48
#define CONV2 48*3*3*128	// number of elements in filters of CONV2 is 48*3*3*128
#define CONV3 128*3*3*128	// number of elements in filters of CONV3 is 128*3*3*128
#define CONV4 128*3*3*256	// number of elements in filters of CONV4 is 128*3*3*256
#define CONV5 256*3*3*192	// number of elements in filters of CONV5 is 256*3*3*192
#define CONV6 192*3*3*192	// number of elements in filters of CONV6 is 192*3*3*192
#define CONV7 192*3*3*128	// number of elements in filters of CONV7 is 192*3*3*128

#ifdef __VIPL_LOG__
#include <ctime>
#endif

#ifdef SDSOC    // for sds_alloc
#include "sds_lib.h"
#endif

void ConvNet::SetUp() {
  stride_h_ = stride_w_ =
      *(int*)(this->hyper_param()->param("stride"));

  // check input and output blob size
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
  this->params().resize(1);
}

void ConvNet::Execute() {
#ifdef __VIPL_LOG__
  double t_start, t_end, scan_time, math_time;
#endif
  // *** Argument *** //
  const bool is_binary = false;
  // *** //

  CheckInput();
  const Blob* const input = this->input_blobs(0);
  const Blob* const weight = this->params(0);
  Blob* const output = this->output_blobs(0);

  int src_num = input->num();   // 经测试，这里src_num总是等于1
  int src_channels = input->channels();
  int src_h = input->height();
  int src_w = input->width();
  int dst_channels = weight->num();
  int kernel_h = weight->height();
  int kernel_w = weight->width();

  LOG(DEBUG) << "input blob: (" <<src_num << "," << src_channels << "," << src_h
    << "," << src_w << ")";

  int dst_h = (src_h - kernel_h) / stride_h_ + 1;	// 纵向移窗数
  int dst_w = (src_w - kernel_w) / stride_w_ + 1;	// 横向移窗数
  int end_h = src_h - kernel_h + 1;
  int end_w = src_w - kernel_w + 1;
  int dst_size = dst_h * dst_w;
  int kernel_size = src_channels * kernel_h * kernel_w;
  const int src_num_offset = src_channels * src_h * src_w;
  
  const int KERNELS = 3*9*9*48;
  const int max_ifm = 128*29*29;
  const int max_fil = 192*256*3*3;
  const int max_ofm = 128*27*27;
  int fac_para;
  switch (dst_channels * kernel_size)
  {
      case CONV1:
          fac_para = 1;
          break;

      default:
          fac_para = 16;
          break;
  }

  // 给硬件函数的输入分配物理连续内存
#ifndef SDSOC
    data_1* const ofmaps_1 = new data_1[48*55*55];
    data_1* const ofmaps_2 = new data_1[max_ofm];
    data_1* const ifmaps_1 = new data_1[3*228*228];
    data_1* const ifmaps_2 = new data_1[max_ifm];
    float* dst_head = new float[dst_size * dst_channels];
#else
    data_1* const ofmaps_1 = (data_1 *)sds_alloc(48*55*55 * sizeof(data_1));
    data_1* const ofmaps_2 = (data_1 *)sds_alloc(max_ofm * sizeof(data_1));
    data_1* const ifmaps_1 = (data_1 *)sds_alloc(3*228*228 * sizeof(data_1));
    data_1* const ifmaps_2 = (data_1 *)sds_alloc(max_ifm * sizeof(data_1));
    float* dst_head = (float *)sds_alloc(dst_size * dst_channels * sizeof(float));
#endif // SDSOC

  const float* src_data = input->data().get();
  float* dst_data = dst_head;
  int didx = 0;
#ifdef __VIPL_LOG__
  scan_time = math_time = 0;
#endif
  for (int sn = 0; sn < src_num; ++sn) {				// 切换input的num维度，但记住src_num在此等于1。
#ifdef __VIPL_LOG__
    t_start = clock();
#endif

    // 精度转换并为ifmaps动态空间分配数据
    switch (dst_channels * kernel_size){
        case CONV1:{
            for(int icp = 0; icp < src_num_offset; icp++){
                ifmaps_1[icp] = static_cast<data_1>(src_data[icp]);
            }
            break;
        }
        default:{
            for(int icp = 0; icp < src_num_offset; icp++){
                ifmaps_2[icp] = static_cast<data_1>(src_data[icp]);
            }
            break;
        }
    }

#ifdef __VIPL_LOG__
    t_end = clock();
    scan_time += t_end - t_start;

    t_start = clock();
#endif

    // 精度转换并为filter动态空间分配数据。
    const float* ptr_temp = weight->data().get();
#ifndef SDSOC
    data_2* const filters_1 = new data_2[48*3*9*9];
    data_2* const filters_2 = new data_2[max_fil];
    switch(dst_channels * kernel_size){
        case CONV1:{
            for(int icp = 0, chan = 0; chan < src_channels/fac_para; chan++){
                for(int fil = 0; fil < dst_channels; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < kernel_h; hidx++){
                            for(int widx = 0; widx < kernel_w; widx++){
                                filters_1[icp] = static_cast<data_2>(ptr_temp[fil*kernel_size + chan*fac_para*kernel_h*kernel_w + cidx*kernel_h*kernel_w + hidx*kernel_w + widx]);
                                icp++;
                            }
                        }
                    }
                }
            }
            break;
        }
        default:{
            for(int icp = 0, chan = 0; chan < src_channels/fac_para; chan++){
                for(int fil = 0; fil < dst_channels; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < kernel_h; hidx++){
                            for(int widx = 0; widx < kernel_w; widx++){
                                filters_2[icp] = static_cast<data_2>(ptr_temp[fil*kernel_size + chan*fac_para*kernel_h*kernel_w + cidx*kernel_h*kernel_w + hidx*kernel_w + widx]);
                                icp++;
                            }
                        }
                    }
                }
            }
            break;
        }
    }
#else
    data_2* const filters_1 = (data_2 *)sds_alloc(48*3*9*9 * sizeof(data_2));
    data_2* const filters_2 = (data_2 *)sds_alloc(max_fil * sizeof(data_2));
    switch(dst_channels * kernel_size){
        case CONV1:{
            for(int icp = 0, chan = 0; chan < src_channels/fac_para; chan++){
                for(int fil = 0; fil < dst_channels; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < kernel_h; hidx++){
                            for(int widx = 0; widx < kernel_w; widx++){
                                filters_1[icp] = static_cast<data_2>(ptr_temp[fil*kernel_size + chan*fac_para*kernel_h*kernel_w + cidx*kernel_h*kernel_w + hidx*kernel_w + widx]);
                                icp++;
                            }
                        }
                    }
                }
            }
            break;
        }
        default:{
            for(int icp = 0, chan = 0; chan < src_channels/fac_para; chan++){
                for(int fil = 0; fil < dst_channels; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < kernel_h; hidx++){
                            for(int widx = 0; widx < kernel_w; widx++){
                                filters_2[icp] = static_cast<data_2>(ptr_temp[fil*kernel_size + chan*fac_para*kernel_h*kernel_w + cidx*kernel_h*kernel_w + hidx*kernel_w + widx]);
                                icp++;
                            }
                        }
                    }
                }
            }
            break;
        }
    }
#endif // SDSOC

    // 开始计算卷积层。
    if( KERNELS == kernel_size * dst_channels){    
        clock_t start_clock, cnt = 0;
        start_clock = clock();
        convolute1(ifmaps_1, filters_1, ofmaps_1, dst_channels, kernel_size);
        cnt = clock() - start_clock;
        std::cout << "matrix_procuct clock = " << cnt << std::endl;
        // 把结果转换回float
        for(int icp = 0; icp < src_num * dst_size * dst_channels; icp++){
            dst_head[icp] = static_cast<float>(ofmaps_1[icp]);
        }
    }     // if
    else{
        clock_t start_clock, cnt = 0;
        start_clock = clock();
        matrix_procuct(ifmaps_2, filters_2, ofmaps_2, dst_channels, kernel_size);
        cnt = clock() - start_clock;
        std::cout << "matrix_procuct clock = " << cnt << std::endl;
        // 把结果转换回float
        for(int icp = 0; icp < src_num * dst_size * dst_channels; icp++){
            dst_head[icp] = static_cast<float>(ofmaps_2[icp]);
        }
    }     //else

#ifndef SDSOC
    delete[] filters_2;
    delete[] filters_1;
#else
    sds_free(filters_2);
    sds_free(filters_1);
#endif // SDSOC


#ifdef __VIPL_LOG__
    t_end = clock();
    math_time += t_end - t_start;
#endif
    dst_data += dst_channels * dst_size;
  } // for sn

#ifdef __VIPL_LOG__
  LOG(INFO) << "scan time: " << scan_time / CLOCKS_PER_SEC * 1000 << "ms";
  LOG(INFO) << "math time: " << math_time / CLOCKS_PER_SEC * 1000 << "ms";
#endif
  output->CopyData(src_num, dst_channels, dst_h, dst_w, dst_head);

#ifndef SDSOC
  //delete[] mat_head;
  delete[] dst_head;
  delete[] ifmaps_2;
  delete[] ifmaps_1;
  delete[] ofmaps_2;
  delete[] ofmaps_1;
#else
  //sds_free(mat_head);
  sds_free(dst_head);
  sds_free(ifmaps_2);
  sds_free(ifmaps_1);
  sds_free(ofmaps_2);
  sds_free(ofmaps_1);
#endif // SDSOC

  LOG(DEBUG) << "output blob: (" << output->num() << "," << output->channels()
    << "," << output->height() << "," << output->width() << ")";
  CheckOutput();
}

REGISTER_NET_CLASS(Conv);
