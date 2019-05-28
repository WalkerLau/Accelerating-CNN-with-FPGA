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

#include "sdsoc.h"
#include "datatype.h"
#include "math_functions.h"

#define CONV1 3*9*9*48	    // number of elements in filters of CONV1 is 3*9*9*48
#define CONV2 48*3*3*128	// number of elements in filters of CONV2 is 48*3*3*128
#define CONV3 128*3*3*128	// number of elements in filters of CONV3 is 128*3*3*128
#define CONV4 128*3*3*256	// number of elements in filters of CONV4 is 128*3*3*256
#define CONV5 256*3*3*192	// number of elements in filters of CONV5 is 256*3*3*192
#define CONV6 192*3*3*192	// number of elements in filters of CONV6 is 192*3*3*192
#define CONV7 192*3*3*128	// number of elements in filters of CONV7 is 192*3*3*128

float simd_dot(const float* x, const float* y, const long& k){	// 注意：simd_dot函数在其他多个文件中也有被引用（条件分支），所以不要更改这个函数的名字
	float inner_prod = 0;
	for(int i = 0; i < k; i++){
		inner_prod += x[i]*y[i];
	}
	return inner_prod;
}

// matrix_procuct的输入分别为：输入数据首地址A、权重数据首地址B、输出数据首地址C、ofmap平面元素数量n、output volume的channel数m、一个filter的元素数量k。。。
void matrix_procuct(const data_1 A[128*29*29], const data_2 B[192*256*3*3], data_1 C[128*27*27], const int m, const int k){

    const int ifmelmt = 128*29*29;
    const int filelmt = 192*256*3*3;
    const int ofmelmt = 128*27*27;

    int fac_para = 16;            // 其余层公用的fac_para。注意共享BRAM的相应尺寸要手动更改，sdsoc不支持数组尺寸为无限定变量。
    //int fac_para_1 = 1;         // 第一层定制的fac_para。注意共享BRAM的相应尺寸要手动更改，sdsoc不支持数组尺寸为无限定变量。

    //data_1 ifmaps1[1][228][228];            // 存第一层ifmaps。
    data_1 ifmaps2[16][29][29];               // 第2~3层共享。
    data_1 ifmaps3[16][15][15];               // 第4~7层共享。
    //data_1 filter1[48][1][9][9];            // 存第一层filter。
    data_2 filter2[256][16][3][3];            // 第2~7层共享。
    data_1 ofmaps[128*27*27];                 // 所有层共享。
    #ifdef SDSOC
        //#pragma HLS array_partition variable=ifmaps1 complete dim=1
        #pragma HLS array_partition variable=ifmaps2 complete dim=1
        #pragma HLS array_partition variable=ifmaps3 complete dim=1
        //#pragma HLS array_partition variable=ifmaps3 cyclic factor=filter_h dim=2
        //#pragma HLS array_partition variable=ifmaps3 cyclic factor=filter_w dim=3
        //#pragma HLS array_partition variable=filter1 complete dim=2
        //#pragma HLS array_partition variable=filter1 complete dim=3
        //#pragma HLS array_partition variable=filter1 complete dim=4
        #pragma HLS array_partition variable=filter2 complete dim=2
        #pragma HLS array_partition variable=filter2 complete dim=3
        #pragma HLS array_partition variable=filter2 complete dim=4
    #endif

    data_1 actor_1;   data_2 actor_2;   // actor机制。

    data_3 a_c1;      data_3 a_c2;

    data_3 a1;        data_3 a2;        data_3 a3;         data_3 a4;

    data_3 ac1;       data_3 ac2;       data_3 ac3;        data_3 ac4;       data_3 ac5;       data_3 ac6;       data_3 ac7;      data_3 ac8;

    data_3 acc1;      data_3 acc2;      data_3 acc3;       data_3 acc4;      data_3 acc5;      data_3 acc6;      data_3 acc7;     data_3 acc8;     data_3 acc9;     data_3 acc10;     data_3 acc11;    data_3 acc12;    data_3 acc13;    data_3 acc14;    data_3 acc15;    data_3 acc16;

    data_3 acce011;   data_3 acce021;   data_3 acce031;    data_3 acce041;   data_3 acce051;   data_3 acce061;   data_3 acce071;  data_3 acce081;  data_3 acce091;  data_3 acce101;   data_3 acce111;  data_3 acce121;  data_3 acce131;  data_3 acce141;  data_3 acce151;  data_3 acce161;
    data_3 acce012;   data_3 acce022;   data_3 acce032;    data_3 acce042;   data_3 acce052;   data_3 acce062;   data_3 acce072;  data_3 acce082;  data_3 acce092;  data_3 acce102;   data_3 acce112;  data_3 acce122;  data_3 acce132;  data_3 acce142;  data_3 acce152;  data_3 acce162;

    data_3 accd011;   data_3 accd021;   data_3 accd031;    data_3 accd041;   data_3 accd051;   data_3 accd061;   data_3 accd071;  data_3 accd081;  data_3 accd091;  data_3 accd101;   data_3 accd111;  data_3 accd121;  data_3 accd131;  data_3 accd141;  data_3 accd151;  data_3 accd161;
    data_3 accd012;   data_3 accd022;   data_3 accd032;    data_3 accd042;   data_3 accd052;   data_3 accd062;   data_3 accd072;  data_3 accd082;  data_3 accd092;  data_3 accd102;   data_3 accd112;  data_3 accd122;  data_3 accd132;  data_3 accd142;  data_3 accd152;  data_3 accd162;
    data_3 accd013;   data_3 accd023;   data_3 accd033;    data_3 accd043;   data_3 accd053;   data_3 accd063;   data_3 accd073;  data_3 accd083;  data_3 accd093;  data_3 accd103;   data_3 accd113;  data_3 accd123;  data_3 accd133;  data_3 accd143;  data_3 accd153;  data_3 accd163;
    data_3 accd014;   data_3 accd024;   data_3 accd034;    data_3 accd044;   data_3 accd054;   data_3 accd064;   data_3 accd074;  data_3 accd084;  data_3 accd094;  data_3 accd104;   data_3 accd114;  data_3 accd124;  data_3 accd134;  data_3 accd144;  data_3 accd154;  data_3 accd164;
 
    data_3 accc011;   data_3 accc021;   data_3 accc031;    data_3 accc041;   data_3 accc051;   data_3 accc061;   data_3 accc071;  data_3 accc081;  data_3 accc091;  data_3 accc101;   data_3 accc111;  data_3 accc121;  data_3 accc131;  data_3 accc141;  data_3 accc151;  data_3 accc161;
    data_3 accc012;   data_3 accc022;   data_3 accc032;    data_3 accc042;   data_3 accc052;   data_3 accc062;   data_3 accc072;  data_3 accc082;  data_3 accc092;  data_3 accc102;   data_3 accc112;  data_3 accc122;  data_3 accc132;  data_3 accc142;  data_3 accc152;  data_3 accc162;
    data_3 accc013;   data_3 accc023;   data_3 accc033;    data_3 accc043;   data_3 accc053;   data_3 accc063;   data_3 accc073;  data_3 accc083;  data_3 accc093;  data_3 accc103;   data_3 accc113;  data_3 accc123;  data_3 accc133;  data_3 accc143;  data_3 accc153;  data_3 accc163;
    data_3 accc014;   data_3 accc024;   data_3 accc034;    data_3 accc044;   data_3 accc054;   data_3 accc064;   data_3 accc074;  data_3 accc084;  data_3 accc094;  data_3 accc104;   data_3 accc114;  data_3 accc124;  data_3 accc134;  data_3 accc144;  data_3 accc154;  data_3 accc164;
    data_3 accc015;   data_3 accc025;   data_3 accc035;    data_3 accc045;   data_3 accc055;   data_3 accc065;   data_3 accc075;  data_3 accc085;  data_3 accc095;  data_3 accc105;   data_3 accc115;  data_3 accc125;  data_3 accc135;  data_3 accc145;  data_3 accc155;  data_3 accc165;
    data_3 accc016;   data_3 accc026;   data_3 accc036;    data_3 accc046;   data_3 accc056;   data_3 accc066;   data_3 accc076;  data_3 accc086;  data_3 accc096;  data_3 accc106;   data_3 accc116;  data_3 accc126;  data_3 accc136;  data_3 accc146;  data_3 accc156;  data_3 accc166;
    data_3 accc017;   data_3 accc027;   data_3 accc037;    data_3 accc047;   data_3 accc057;   data_3 accc067;   data_3 accc077;  data_3 accc087;  data_3 accc097;  data_3 accc107;   data_3 accc117;  data_3 accc127;  data_3 accc137;  data_3 accc147;  data_3 accc157;  data_3 accc167;
    data_3 accc018;   data_3 accc028;   data_3 accc038;    data_3 accc048;   data_3 accc058;   data_3 accc068;   data_3 accc078;  data_3 accc088;  data_3 accc098;  data_3 accc108;   data_3 accc118;  data_3 accc128;  data_3 accc138;  data_3 accc148;  data_3 accc158;  data_3 accc168;
    data_3 accc019;   data_3 accc029;   data_3 accc039;    data_3 accc049;   data_3 accc059;   data_3 accc069;   data_3 accc079;  data_3 accc089;  data_3 accc099;  data_3 accc109;   data_3 accc119;  data_3 accc129;  data_3 accc139;  data_3 accc149;  data_3 accc159;  data_3 accc169;

    switch (m * k)
    {
        case CONV2:{
            const int src_c = 48;
            const int src_h = 29;
            const int src_w = 29;
            const int stride = 1;
            const int filter_size = 48*3*3;
            const int filter_h = 3;
            const int filter_w = 3;
            const int dst_c = 128;
            const int dst_h = 27;
            const int dst_w = 27;
            const int end_h = src_h - filter_h + 1;
            const int end_w = src_w - filter_w + 1;

            // 给ofmaps进行初始化
            for(int i = 0; i < dst_c*dst_h*dst_w; i++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                ofmaps[i] = 0;
            }

            for (int icp_ifm = 0, icp_fil = 0,  chan = 0; chan < src_c; chan += fac_para){        // 一个filter的channel维度。其中若chan+=4代表4个filter channel在并行。
                // 用local memory ifmaps2来存储输入。
                for(int cidx = 0; cidx < fac_para; cidx++){
                    for(int hidx = 0; hidx < src_h; hidx++){
                        for(int widx = 0; widx < src_w; widx++){
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            #endif
                            ifmaps2[cidx][hidx][widx] = A[icp_ifm];
                            icp_ifm++;
                        }
                    }
                }
                // actor机制
                if(icp_ifm == src_c*src_h*src_w){
                    for(;icp_ifm < ifmelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif                    
                        actor_1 = A[icp_ifm];
                        icp_ifm++;                              
                    }
                    //std::cout << "actor on, the last icp_ifm = " << icp_ifm - 1 << std::endl;  
                }   
                // 用local memory来存储filter。
                for(int fil = 0; fil < dst_c; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < filter_h; hidx++){
                            for(int widx = 0; widx < filter_w; widx++){
                                #ifdef SDSOC
                                #pragma HLS PIPELINE
                                #endif
                                filter2[fil][cidx][hidx][widx] = B[icp_fil];
                                icp_fil++;
                            }
                        }
                    }
                }
                // actor机制
                if(icp_fil == dst_c*src_c*filter_h*filter_w){
                    for(;icp_fil < filelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif
                        actor_2 = B[icp_fil];
                        icp_fil++;
                    }
                    //std::cout << "actor on, the last icp_fil = " << icp_fil - 1 << std::endl;  
                }  
                // 移窗
                for(int sh = 0; sh < end_h; sh += stride){                  // 纵向移窗。
                    for (int sw = 0; sw < end_w; sw += stride){             // 横向移窗。
                        int ofm_idx = sh/stride * dst_w + sw/stride;
                        for(int fil = 0; fil < dst_c; fil++){      // 切换filter。
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            //#pragma HLS dependence array inter false
                            #endif
                            accc011 = ifmaps2[0][sh + 0][sw + 0] * filter2[fil][0][0][0];
                            accc012 = ifmaps2[0][sh + 0][sw + 1] * filter2[fil][0][0][1];
                            accd011 =   accc011 + accc012;
                            accc013 = ifmaps2[0][sh + 0][sw + 2] * filter2[fil][0][0][2];
                            accc014 = ifmaps2[0][sh + 1][sw + 0] * filter2[fil][0][1][0];
                            accd012 =   accc013 + accc014;
                            acce011 =       accd011 + accd012;
                            accc015 = ifmaps2[0][sh + 1][sw + 1] * filter2[fil][0][1][1];
                            accc016 = ifmaps2[0][sh + 1][sw + 2] * filter2[fil][0][1][2];
                            accd013 =   accc015 + accc016;
                            accc017 = ifmaps2[0][sh + 2][sw + 0] * filter2[fil][0][2][0];
                            accc018 = ifmaps2[0][sh + 2][sw + 1] * filter2[fil][0][2][1];
                            accd014 =   accc017 + accc018;
                            acce012 =       accd013 + accd014;
                            accc019 = ifmaps2[0][sh + 2][sw + 2] * filter2[fil][0][2][2];
                            acc1 =  acce011 + acce012 + accc019;

                            accc021 = ifmaps2[1][sh + 0][sw + 0] * filter2[fil][1][0][0];
                            accc022 = ifmaps2[1][sh + 0][sw + 1] * filter2[fil][1][0][1];
                            accd021 =   accc021 + accc022;
                            accc023 = ifmaps2[1][sh + 0][sw + 2] * filter2[fil][1][0][2];
                            accc024 = ifmaps2[1][sh + 1][sw + 0] * filter2[fil][1][1][0];
                            accd022 =   accc023 + accc024;
                            acce021 =       accd021 + accd022;
                            accc025 = ifmaps2[1][sh + 1][sw + 1] * filter2[fil][1][1][1];
                            accc026 = ifmaps2[1][sh + 1][sw + 2] * filter2[fil][1][1][2];
                            accd023 =   accc025 + accc026;
                            accc027 = ifmaps2[1][sh + 2][sw + 0] * filter2[fil][1][2][0];
                            accc028 = ifmaps2[1][sh + 2][sw + 1] * filter2[fil][1][2][1];
                            accd024 =   accc027 + accc028;
                            acce022 =       accd023 + accd024;
                            accc029 = ifmaps2[1][sh + 2][sw + 2] * filter2[fil][1][2][2];
                            acc2 =  acce021 + acce022 + accc029;

                            ac1 = acc1 + acc2;

                            accc031 = ifmaps2[2][sh + 0][sw + 0] * filter2[fil][2][0][0];
                            accc032 = ifmaps2[2][sh + 0][sw + 1] * filter2[fil][2][0][1];
                            accd031 =   accc031 + accc032;
                            accc033 = ifmaps2[2][sh + 0][sw + 2] * filter2[fil][2][0][2];
                            accc034 = ifmaps2[2][sh + 1][sw + 0] * filter2[fil][2][1][0];
                            accd032 =   accc033 + accc034;
                            acce031 =       accd031 + accd032;
                            accc035 = ifmaps2[2][sh + 1][sw + 1] * filter2[fil][2][1][1];
                            accc036 = ifmaps2[2][sh + 1][sw + 2] * filter2[fil][2][1][2];
                            accd033 =   accc035 + accc036;
                            accc037 = ifmaps2[2][sh + 2][sw + 0] * filter2[fil][2][2][0];
                            accc038 = ifmaps2[2][sh + 2][sw + 1] * filter2[fil][2][2][1];
                            accd034 =   accc037 + accc038;
                            acce032 =       accd033 + accd034;
                            accc039 = ifmaps2[2][sh + 2][sw + 2] * filter2[fil][2][2][2];
                            acc3 =  acce031 + acce032 + accc039;

                            accc041 = ifmaps2[3][sh + 0][sw + 0] * filter2[fil][3][0][0];
                            accc042 = ifmaps2[3][sh + 0][sw + 1] * filter2[fil][3][0][1];
                            accd041 =   accc041 + accc042;
                            accc043 = ifmaps2[3][sh + 0][sw + 2] * filter2[fil][3][0][2];
                            accc044 = ifmaps2[3][sh + 1][sw + 0] * filter2[fil][3][1][0];
                            accd042 =   accc043 + accc044;
                            acce041 =       accd041 + accd042;
                            accc045 = ifmaps2[3][sh + 1][sw + 1] * filter2[fil][3][1][1];
                            accc046 = ifmaps2[3][sh + 1][sw + 2] * filter2[fil][3][1][2];
                            accd043 =   accc045 + accc046;
                            accc047 = ifmaps2[3][sh + 2][sw + 0] * filter2[fil][3][2][0];
                            accc048 = ifmaps2[3][sh + 2][sw + 1] * filter2[fil][3][2][1];
                            accd044 =   accc047 + accc048;
                            acce042 =       accd043 + accd044;
                            accc049 = ifmaps2[3][sh + 2][sw + 2] * filter2[fil][3][2][2];
                            acc4 =  acce041 + acce042 + accc049;

                            ac2 = acc3 +acc4;
                            a1 = ac1 + ac2;

                            accc051 = ifmaps2[4][sh + 0][sw + 0] * filter2[fil][4][0][0];
                            accc052 = ifmaps2[4][sh + 0][sw + 1] * filter2[fil][4][0][1];
                            accd051 =   accc051 + accc052;
                            accc053 = ifmaps2[4][sh + 0][sw + 2] * filter2[fil][4][0][2];
                            accc054 = ifmaps2[4][sh + 1][sw + 0] * filter2[fil][4][1][0];
                            accd052 =   accc053 + accc054;
                            acce051 =       accd051 + accd052;
                            accc055 = ifmaps2[4][sh + 1][sw + 1] * filter2[fil][4][1][1];
                            accc056 = ifmaps2[4][sh + 1][sw + 2] * filter2[fil][4][1][2];
                            accd053 =   accc055 + accc056;
                            accc057 = ifmaps2[4][sh + 2][sw + 0] * filter2[fil][4][2][0];
                            accc058 = ifmaps2[4][sh + 2][sw + 1] * filter2[fil][4][2][1];
                            accd054 =   accc057 + accc058;
                            acce052 =       accd053 + accd054;
                            accc059 = ifmaps2[4][sh + 2][sw + 2] * filter2[fil][4][2][2];
                            acc5 =  acce051 + acce052 + accc059;

                            accc061 = ifmaps2[5][sh + 0][sw + 0] * filter2[fil][5][0][0];
                            accc062 = ifmaps2[5][sh + 0][sw + 1] * filter2[fil][5][0][1];
                            accd061 =   accc061 + accc062;
                            accc063 = ifmaps2[5][sh + 0][sw + 2] * filter2[fil][5][0][2];
                            accc064 = ifmaps2[5][sh + 1][sw + 0] * filter2[fil][5][1][0];
                            accd062 =   accc063 + accc064;
                            acce061 =       accd061 + accd062;
                            accc065 = ifmaps2[5][sh + 1][sw + 1] * filter2[fil][5][1][1];
                            accc066 = ifmaps2[5][sh + 1][sw + 2] * filter2[fil][5][1][2];
                            accd063 =   accc065 + accc066;
                            accc067 = ifmaps2[5][sh + 2][sw + 0] * filter2[fil][5][2][0];
                            accc068 = ifmaps2[5][sh + 2][sw + 1] * filter2[fil][5][2][1];
                            accd064 =   accc067 + accc068;
                            acce062 =       accd063 + accd064;
                            accc069 = ifmaps2[5][sh + 2][sw + 2] * filter2[fil][5][2][2];
                            acc6 =  acce061 + acce062 + accc069;

                            ac3 = acc5 + acc6;

                            accc071 = ifmaps2[6][sh + 0][sw + 0] * filter2[fil][6][0][0];
                            accc072 = ifmaps2[6][sh + 0][sw + 1] * filter2[fil][6][0][1];
                            accd071 =   accc071 + accc072;
                            accc073 = ifmaps2[6][sh + 0][sw + 2] * filter2[fil][6][0][2];
                            accc074 = ifmaps2[6][sh + 1][sw + 0] * filter2[fil][6][1][0];
                            accd072 =   accc073 + accc074;
                            acce071 =       accd071 + accd072;
                            accc075 = ifmaps2[6][sh + 1][sw + 1] * filter2[fil][6][1][1];
                            accc076 = ifmaps2[6][sh + 1][sw + 2] * filter2[fil][6][1][2];
                            accd073 =   accc075 + accc076;
                            accc077 = ifmaps2[6][sh + 2][sw + 0] * filter2[fil][6][2][0];
                            accc078 = ifmaps2[6][sh + 2][sw + 1] * filter2[fil][6][2][1];
                            accd074 =   accc077 + accc078;
                            acce072 =       accd073 + accd074;
                            accc079 = ifmaps2[6][sh + 2][sw + 2] * filter2[fil][6][2][2];
                            acc7 =  acce071 + acce072 + accc079;

                            accc081 = ifmaps2[7][sh + 0][sw + 0] * filter2[fil][7][0][0];
                            accc082 = ifmaps2[7][sh + 0][sw + 1] * filter2[fil][7][0][1];
                            accd081 =   accc081 + accc082;
                            accc083 = ifmaps2[7][sh + 0][sw + 2] * filter2[fil][7][0][2];
                            accc084 = ifmaps2[7][sh + 1][sw + 0] * filter2[fil][7][1][0];
                            accd082 =   accc083 + accc084;
                            acce081 =       accd081 + accd082;
                            accc085 = ifmaps2[7][sh + 1][sw + 1] * filter2[fil][7][1][1];
                            accc086 = ifmaps2[7][sh + 1][sw + 2] * filter2[fil][7][1][2];
                            accd083 =   accc085 + accc086;
                            accc087 = ifmaps2[7][sh + 2][sw + 0] * filter2[fil][7][2][0];
                            accc088 = ifmaps2[7][sh + 2][sw + 1] * filter2[fil][7][2][1];
                            accd084 =   accc087 + accc088;
                            acce082 =       accd083 + accd084;
                            accc089 = ifmaps2[7][sh + 2][sw + 2] * filter2[fil][7][2][2];
                            acc8 =  acce081 + acce082 + accc089;

                            ac4 = acc7 + acc8;
                            a2 = ac3 + ac4;
                            a_c1 = a1 + a2;

                            accc091 = ifmaps2[8][sh + 0][sw + 0] * filter2[fil][8][0][0];
                            accc092 = ifmaps2[8][sh + 0][sw + 1] * filter2[fil][8][0][1];
                            accd091 =   accc091 + accc092;
                            accc093 = ifmaps2[8][sh + 0][sw + 2] * filter2[fil][8][0][2];
                            accc094 = ifmaps2[8][sh + 1][sw + 0] * filter2[fil][8][1][0];
                            accd092 =   accc093 + accc094;
                            acce091 =       accd091 + accd092;
                            accc095 = ifmaps2[8][sh + 1][sw + 1] * filter2[fil][8][1][1];
                            accc096 = ifmaps2[8][sh + 1][sw + 2] * filter2[fil][8][1][2];
                            accd093 =   accc095 + accc096;
                            accc097 = ifmaps2[8][sh + 2][sw + 0] * filter2[fil][8][2][0];
                            accc098 = ifmaps2[8][sh + 2][sw + 1] * filter2[fil][8][2][1];
                            accd094 =   accc097 + accc098;
                            acce092 =       accd093 + accd094;
                            accc099 = ifmaps2[8][sh + 2][sw + 2] * filter2[fil][8][2][2];
                            acc9 =  acce091 + acce092 + accc099;

                            accc101 = ifmaps2[9][sh + 0][sw + 0] * filter2[fil][9][0][0];
                            accc102 = ifmaps2[9][sh + 0][sw + 1] * filter2[fil][9][0][1];
                            accd101 =   accc101 + accc102;
                            accc103 = ifmaps2[9][sh + 0][sw + 2] * filter2[fil][9][0][2];
                            accc104 = ifmaps2[9][sh + 1][sw + 0] * filter2[fil][9][1][0];
                            accd102 =   accc103 + accc104;
                            acce101 =       accd101 + accd102;
                            accc105 = ifmaps2[9][sh + 1][sw + 1] * filter2[fil][9][1][1];
                            accc106 = ifmaps2[9][sh + 1][sw + 2] * filter2[fil][9][1][2];
                            accd103 =   accc105 + accc106;
                            accc107 = ifmaps2[9][sh + 2][sw + 0] * filter2[fil][9][2][0];
                            accc108 = ifmaps2[9][sh + 2][sw + 1] * filter2[fil][9][2][1];
                            accd104 =   accc107 + accc108;
                            acce102 =       accd103 + accd104;
                            accc109 = ifmaps2[9][sh + 2][sw + 2] * filter2[fil][9][2][2];
                            acc10 =  acce101 + acce102 + accc109;

                            ac5 = acc9 + acc10;

                            accc111 = ifmaps2[10][sh + 0][sw + 0] * filter2[fil][10][0][0];
                            accc112 = ifmaps2[10][sh + 0][sw + 1] * filter2[fil][10][0][1];
                            accd111 =   accc111 + accc112;
                            accc113 = ifmaps2[10][sh + 0][sw + 2] * filter2[fil][10][0][2];
                            accc114 = ifmaps2[10][sh + 1][sw + 0] * filter2[fil][10][1][0];
                            accd112 =   accc113 + accc114;
                            acce111 =       accd111 + accd112;
                            accc115 = ifmaps2[10][sh + 1][sw + 1] * filter2[fil][10][1][1];
                            accc116 = ifmaps2[10][sh + 1][sw + 2] * filter2[fil][10][1][2];
                            accd113 =   accc115 + accc116;
                            accc117 = ifmaps2[10][sh + 2][sw + 0] * filter2[fil][10][2][0];
                            accc118 = ifmaps2[10][sh + 2][sw + 1] * filter2[fil][10][2][1];
                            accd114 =   accc117 + accc118;
                            acce112 =       accd113 + accd114;
                            accc119 = ifmaps2[10][sh + 2][sw + 2] * filter2[fil][10][2][2];
                            acc11 =  acce111 + acce112 + accc119;

                            accc121 = ifmaps2[11][sh + 0][sw + 0] * filter2[fil][11][0][0];
                            accc122 = ifmaps2[11][sh + 0][sw + 1] * filter2[fil][11][0][1];
                            accd121 =   accc121 + accc122;
                            accc123 = ifmaps2[11][sh + 0][sw + 2] * filter2[fil][11][0][2];
                            accc124 = ifmaps2[11][sh + 1][sw + 0] * filter2[fil][11][1][0];
                            accd122 =   accc123 + accc124;
                            acce121 =       accd121 + accd122;
                            accc125 = ifmaps2[11][sh + 1][sw + 1] * filter2[fil][11][1][1];
                            accc126 = ifmaps2[11][sh + 1][sw + 2] * filter2[fil][11][1][2];
                            accd123 =   accc125 + accc126;
                            accc127 = ifmaps2[11][sh + 2][sw + 0] * filter2[fil][11][2][0];
                            accc128 = ifmaps2[11][sh + 2][sw + 1] * filter2[fil][11][2][1];
                            accd124 =   accc127 + accc128;
                            acce122 =       accd123 + accd124;
                            accc129 = ifmaps2[11][sh + 2][sw + 2] * filter2[fil][11][2][2];
                            acc12 =  acce121 + acce122 + accc129;

                            ac6 = acc11 + acc12;
                            a3 = ac5 + ac6;

                            accc131 = ifmaps2[12][sh + 0][sw + 0] * filter2[fil][12][0][0];
                            accc132 = ifmaps2[12][sh + 0][sw + 1] * filter2[fil][12][0][1];
                            accd131 =   accc131 + accc132;
                            accc133 = ifmaps2[12][sh + 0][sw + 2] * filter2[fil][12][0][2];
                            accc134 = ifmaps2[12][sh + 1][sw + 0] * filter2[fil][12][1][0];
                            accd132 =   accc133 + accc134;
                            acce131 =       accd131 + accd132;
                            accc135 = ifmaps2[12][sh + 1][sw + 1] * filter2[fil][12][1][1];
                            accc136 = ifmaps2[12][sh + 1][sw + 2] * filter2[fil][12][1][2];
                            accd133 =   accc135 + accc136;
                            accc137 = ifmaps2[12][sh + 2][sw + 0] * filter2[fil][12][2][0];
                            accc138 = ifmaps2[12][sh + 2][sw + 1] * filter2[fil][12][2][1];
                            accd134 =   accc137 + accc138;
                            acce132 =       accd133 + accd134;
                            accc139 = ifmaps2[12][sh + 2][sw + 2] * filter2[fil][12][2][2];
                            acc13 =  acce131 + acce132 + accc139;

                            accc141 = ifmaps2[13][sh + 0][sw + 0] * filter2[fil][13][0][0];
                            accc142 = ifmaps2[13][sh + 0][sw + 1] * filter2[fil][13][0][1];
                            accd141 =   accc141 + accc142;
                            accc143 = ifmaps2[13][sh + 0][sw + 2] * filter2[fil][13][0][2];
                            accc144 = ifmaps2[13][sh + 1][sw + 0] * filter2[fil][13][1][0];
                            accd142 =   accc143 + accc144;
                            acce141 =       accd141 + accd142;
                            accc145 = ifmaps2[13][sh + 1][sw + 1] * filter2[fil][13][1][1];
                            accc146 = ifmaps2[13][sh + 1][sw + 2] * filter2[fil][13][1][2];
                            accd143 =   accc145 + accc146;
                            accc147 = ifmaps2[13][sh + 2][sw + 0] * filter2[fil][13][2][0];
                            accc148 = ifmaps2[13][sh + 2][sw + 1] * filter2[fil][13][2][1];
                            accd144 =   accc147 + accc148;
                            acce142 =       accd143 + accd144;
                            accc149 = ifmaps2[13][sh + 2][sw + 2] * filter2[fil][13][2][2];
                            acc14 =  acce141 + acce142 + accc149;

                            ac7 = acc13 + acc14;

                            accc151 = ifmaps2[14][sh + 0][sw + 0] * filter2[fil][14][0][0];
                            accc152 = ifmaps2[14][sh + 0][sw + 1] * filter2[fil][14][0][1];
                            accd151 =   accc151 + accc152;
                            accc153 = ifmaps2[14][sh + 0][sw + 2] * filter2[fil][14][0][2];
                            accc154 = ifmaps2[14][sh + 1][sw + 0] * filter2[fil][14][1][0];
                            accd152 =   accc153 + accc154;
                            acce151 =       accd151 + accd152;
                            accc155 = ifmaps2[14][sh + 1][sw + 1] * filter2[fil][14][1][1];
                            accc156 = ifmaps2[14][sh + 1][sw + 2] * filter2[fil][14][1][2];
                            accd153 =   accc155 + accc156;
                            accc157 = ifmaps2[14][sh + 2][sw + 0] * filter2[fil][14][2][0];
                            accc158 = ifmaps2[14][sh + 2][sw + 1] * filter2[fil][14][2][1];
                            accd154 =   accc157 + accc158;
                            acce152 =       accd153 + accd154;
                            accc159 = ifmaps2[14][sh + 2][sw + 2] * filter2[fil][14][2][2];
                            acc15 =  acce151 + acce152 + accc159;

                            accc161 = ifmaps2[15][sh + 0][sw + 0] * filter2[fil][15][0][0];
                            accc162 = ifmaps2[15][sh + 0][sw + 1] * filter2[fil][15][0][1];
                            accd161 =   accc161 + accc162;
                            accc163 = ifmaps2[15][sh + 0][sw + 2] * filter2[fil][15][0][2];
                            accc164 = ifmaps2[15][sh + 1][sw + 0] * filter2[fil][15][1][0];
                            accd162 =   accc163 + accc164;
                            acce161 =       accd161 + accd162;
                            accc165 = ifmaps2[15][sh + 1][sw + 1] * filter2[fil][15][1][1];
                            accc166 = ifmaps2[15][sh + 1][sw + 2] * filter2[fil][15][1][2];
                            accd163 =   accc165 + accc166;
                            accc167 = ifmaps2[15][sh + 2][sw + 0] * filter2[fil][15][2][0];
                            accc168 = ifmaps2[15][sh + 2][sw + 1] * filter2[fil][15][2][1];
                            accd164 =   accc167 + accc168;
                            acce162 =       accd163 + accd164;
                            accc169 = ifmaps2[15][sh + 2][sw + 2] * filter2[fil][15][2][2];
                            acc16 =  acce161 + acce162 + accc169;

                            ac8 = acc15 + acc16;
                            a4 = ac7 + ac8;
                            a_c2 = a3 + a4;

                            // a1 + a2 + a3 + a4;   // a_c1 + a_c2;
                            //acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + + acc10  + acc11  + acc12  + acc13  + acc14  + acc15  + acc16;
                            ofmaps[ofm_idx] += a_c1 + a_c2;
                            ofm_idx += dst_h * dst_w;


                        }   // for fil
                    }   // for sw
                }   // for sh
            }   // for chan

            int icp_ofm = 0;
            for(; icp_ofm < dst_c*dst_h*dst_w; icp_ofm++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                C[icp_ofm] = ofmaps[icp_ofm];
            }
            // actor机制
            if(icp_ofm == dst_c*dst_h*dst_w){
                for(;icp_ofm < ofmelmt;){
                    #ifdef SDSOC
                    #pragma HLS PIPELINE
                    #endif
                    C[icp_ofm] = 0;
                    icp_ofm++;
                }
                //std::cout << "actor on, the last icp_ofm = " << icp_ofm - 1 << std::endl;
            }

            break;
        }    

        case CONV3:{
            const int src_c = 128;
            const int src_h = 29;
            const int src_w = 29;
            const int stride = 1;
            const int filter_size = 128*3*3;
            const int filter_num = 128;
            const int filter_h = 3;
            const int filter_w = 3;
            const int dst_c = 128;
            const int dst_h = 27;
            const int dst_w = 27;
            const int end_h = src_h - filter_h + 1;
            const int end_w = src_w - filter_w + 1;

            // 给ofmaps进行初始化
            for(int i = 0; i < dst_c*dst_h*dst_w; i++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                ofmaps[i] = 0;
            }

            for (int icp_ifm = 0, icp_fil = 0,  chan = 0; chan < src_c; chan += fac_para){        // 一个filter的channel维度。其中若chan+=4代表4个filter channel在并行。
                // 用local memory ifmaps2来存储输入。
                for(int cidx = 0; cidx < fac_para; cidx++){
                    for(int hidx = 0; hidx < src_h; hidx++){
                        for(int widx = 0; widx < src_w; widx++){
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            #endif
                            ifmaps2[cidx][hidx][widx] = A[icp_ifm];
                            icp_ifm++;
                        }
                    }
                }
                // actor机制
                if(icp_ifm == src_c*src_h*src_w){
                    for(;icp_ifm < ifmelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif                    
                        actor_1 = A[icp_ifm];
                        icp_ifm++;                              
                    }
                    //std::cout << "actor on, the last icp_ifm = " << icp_ifm - 1 << std::endl;  
                }   
                // 用local memory来存储filter。
                for(int fil = 0; fil < dst_c; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < filter_h; hidx++){
                            for(int widx = 0; widx < filter_w; widx++){
                                #ifdef SDSOC
                                #pragma HLS PIPELINE
                                #endif
                                filter2[fil][cidx][hidx][widx] = B[icp_fil];
                                icp_fil++;
                            }
                        }
                    }
                }
                // actor机制
                if(icp_fil == dst_c*src_c*filter_h*filter_w){
                    for(;icp_fil < filelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif
                        actor_2 = B[icp_fil];
                        icp_fil++;
                    }
                    //std::cout << "actor on, the last icp_fil = " << icp_fil - 1 << std::endl;  
                }  
                // 移窗
                for(int sh = 0; sh < end_h; sh += stride){                  // 纵向移窗。
                    for (int sw = 0; sw < end_w; sw += stride){             // 横向移窗。
                        int ofm_idx = sh/stride * dst_w + sw/stride;
                        for(int fil = 0; fil < dst_c; fil++){      // 切换filter。
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            //#pragma HLS dependence array inter false
                            #endif
                            accc011 = ifmaps2[0][sh + 0][sw + 0] * filter2[fil][0][0][0];
                            accc012 = ifmaps2[0][sh + 0][sw + 1] * filter2[fil][0][0][1];
                            accd011 =   accc011 + accc012;
                            accc013 = ifmaps2[0][sh + 0][sw + 2] * filter2[fil][0][0][2];
                            accc014 = ifmaps2[0][sh + 1][sw + 0] * filter2[fil][0][1][0];
                            accd012 =   accc013 + accc014;
                            acce011 =       accd011 + accd012;
                            accc015 = ifmaps2[0][sh + 1][sw + 1] * filter2[fil][0][1][1];
                            accc016 = ifmaps2[0][sh + 1][sw + 2] * filter2[fil][0][1][2];
                            accd013 =   accc015 + accc016;
                            accc017 = ifmaps2[0][sh + 2][sw + 0] * filter2[fil][0][2][0];
                            accc018 = ifmaps2[0][sh + 2][sw + 1] * filter2[fil][0][2][1];
                            accd014 =   accc017 + accc018;
                            acce012 =       accd013 + accd014;
                            accc019 = ifmaps2[0][sh + 2][sw + 2] * filter2[fil][0][2][2];
                            acc1 =  acce011 + acce012 + accc019;

                            accc021 = ifmaps2[1][sh + 0][sw + 0] * filter2[fil][1][0][0];
                            accc022 = ifmaps2[1][sh + 0][sw + 1] * filter2[fil][1][0][1];
                            accd021 =   accc021 + accc022;
                            accc023 = ifmaps2[1][sh + 0][sw + 2] * filter2[fil][1][0][2];
                            accc024 = ifmaps2[1][sh + 1][sw + 0] * filter2[fil][1][1][0];
                            accd022 =   accc023 + accc024;
                            acce021 =       accd021 + accd022;
                            accc025 = ifmaps2[1][sh + 1][sw + 1] * filter2[fil][1][1][1];
                            accc026 = ifmaps2[1][sh + 1][sw + 2] * filter2[fil][1][1][2];
                            accd023 =   accc025 + accc026;
                            accc027 = ifmaps2[1][sh + 2][sw + 0] * filter2[fil][1][2][0];
                            accc028 = ifmaps2[1][sh + 2][sw + 1] * filter2[fil][1][2][1];
                            accd024 =   accc027 + accc028;
                            acce022 =       accd023 + accd024;
                            accc029 = ifmaps2[1][sh + 2][sw + 2] * filter2[fil][1][2][2];
                            acc2 =  acce021 + acce022 + accc029;

                            ac1 = acc1 + acc2;

                            accc031 = ifmaps2[2][sh + 0][sw + 0] * filter2[fil][2][0][0];
                            accc032 = ifmaps2[2][sh + 0][sw + 1] * filter2[fil][2][0][1];
                            accd031 =   accc031 + accc032;
                            accc033 = ifmaps2[2][sh + 0][sw + 2] * filter2[fil][2][0][2];
                            accc034 = ifmaps2[2][sh + 1][sw + 0] * filter2[fil][2][1][0];
                            accd032 =   accc033 + accc034;
                            acce031 =       accd031 + accd032;
                            accc035 = ifmaps2[2][sh + 1][sw + 1] * filter2[fil][2][1][1];
                            accc036 = ifmaps2[2][sh + 1][sw + 2] * filter2[fil][2][1][2];
                            accd033 =   accc035 + accc036;
                            accc037 = ifmaps2[2][sh + 2][sw + 0] * filter2[fil][2][2][0];
                            accc038 = ifmaps2[2][sh + 2][sw + 1] * filter2[fil][2][2][1];
                            accd034 =   accc037 + accc038;
                            acce032 =       accd033 + accd034;
                            accc039 = ifmaps2[2][sh + 2][sw + 2] * filter2[fil][2][2][2];
                            acc3 =  acce031 + acce032 + accc039;

                            accc041 = ifmaps2[3][sh + 0][sw + 0] * filter2[fil][3][0][0];
                            accc042 = ifmaps2[3][sh + 0][sw + 1] * filter2[fil][3][0][1];
                            accd041 =   accc041 + accc042;
                            accc043 = ifmaps2[3][sh + 0][sw + 2] * filter2[fil][3][0][2];
                            accc044 = ifmaps2[3][sh + 1][sw + 0] * filter2[fil][3][1][0];
                            accd042 =   accc043 + accc044;
                            acce041 =       accd041 + accd042;
                            accc045 = ifmaps2[3][sh + 1][sw + 1] * filter2[fil][3][1][1];
                            accc046 = ifmaps2[3][sh + 1][sw + 2] * filter2[fil][3][1][2];
                            accd043 =   accc045 + accc046;
                            accc047 = ifmaps2[3][sh + 2][sw + 0] * filter2[fil][3][2][0];
                            accc048 = ifmaps2[3][sh + 2][sw + 1] * filter2[fil][3][2][1];
                            accd044 =   accc047 + accc048;
                            acce042 =       accd043 + accd044;
                            accc049 = ifmaps2[3][sh + 2][sw + 2] * filter2[fil][3][2][2];
                            acc4 =  acce041 + acce042 + accc049;

                            ac2 = acc3 +acc4;
                            a1 = ac1 + ac2;

                            accc051 = ifmaps2[4][sh + 0][sw + 0] * filter2[fil][4][0][0];
                            accc052 = ifmaps2[4][sh + 0][sw + 1] * filter2[fil][4][0][1];
                            accd051 =   accc051 + accc052;
                            accc053 = ifmaps2[4][sh + 0][sw + 2] * filter2[fil][4][0][2];
                            accc054 = ifmaps2[4][sh + 1][sw + 0] * filter2[fil][4][1][0];
                            accd052 =   accc053 + accc054;
                            acce051 =       accd051 + accd052;
                            accc055 = ifmaps2[4][sh + 1][sw + 1] * filter2[fil][4][1][1];
                            accc056 = ifmaps2[4][sh + 1][sw + 2] * filter2[fil][4][1][2];
                            accd053 =   accc055 + accc056;
                            accc057 = ifmaps2[4][sh + 2][sw + 0] * filter2[fil][4][2][0];
                            accc058 = ifmaps2[4][sh + 2][sw + 1] * filter2[fil][4][2][1];
                            accd054 =   accc057 + accc058;
                            acce052 =       accd053 + accd054;
                            accc059 = ifmaps2[4][sh + 2][sw + 2] * filter2[fil][4][2][2];
                            acc5 =  acce051 + acce052 + accc059;

                            accc061 = ifmaps2[5][sh + 0][sw + 0] * filter2[fil][5][0][0];
                            accc062 = ifmaps2[5][sh + 0][sw + 1] * filter2[fil][5][0][1];
                            accd061 =   accc061 + accc062;
                            accc063 = ifmaps2[5][sh + 0][sw + 2] * filter2[fil][5][0][2];
                            accc064 = ifmaps2[5][sh + 1][sw + 0] * filter2[fil][5][1][0];
                            accd062 =   accc063 + accc064;
                            acce061 =       accd061 + accd062;
                            accc065 = ifmaps2[5][sh + 1][sw + 1] * filter2[fil][5][1][1];
                            accc066 = ifmaps2[5][sh + 1][sw + 2] * filter2[fil][5][1][2];
                            accd063 =   accc065 + accc066;
                            accc067 = ifmaps2[5][sh + 2][sw + 0] * filter2[fil][5][2][0];
                            accc068 = ifmaps2[5][sh + 2][sw + 1] * filter2[fil][5][2][1];
                            accd064 =   accc067 + accc068;
                            acce062 =       accd063 + accd064;
                            accc069 = ifmaps2[5][sh + 2][sw + 2] * filter2[fil][5][2][2];
                            acc6 =  acce061 + acce062 + accc069;

                            ac3 = acc5 + acc6;

                            accc071 = ifmaps2[6][sh + 0][sw + 0] * filter2[fil][6][0][0];
                            accc072 = ifmaps2[6][sh + 0][sw + 1] * filter2[fil][6][0][1];
                            accd071 =   accc071 + accc072;
                            accc073 = ifmaps2[6][sh + 0][sw + 2] * filter2[fil][6][0][2];
                            accc074 = ifmaps2[6][sh + 1][sw + 0] * filter2[fil][6][1][0];
                            accd072 =   accc073 + accc074;
                            acce071 =       accd071 + accd072;
                            accc075 = ifmaps2[6][sh + 1][sw + 1] * filter2[fil][6][1][1];
                            accc076 = ifmaps2[6][sh + 1][sw + 2] * filter2[fil][6][1][2];
                            accd073 =   accc075 + accc076;
                            accc077 = ifmaps2[6][sh + 2][sw + 0] * filter2[fil][6][2][0];
                            accc078 = ifmaps2[6][sh + 2][sw + 1] * filter2[fil][6][2][1];
                            accd074 =   accc077 + accc078;
                            acce072 =       accd073 + accd074;
                            accc079 = ifmaps2[6][sh + 2][sw + 2] * filter2[fil][6][2][2];
                            acc7 =  acce071 + acce072 + accc079;

                            accc081 = ifmaps2[7][sh + 0][sw + 0] * filter2[fil][7][0][0];
                            accc082 = ifmaps2[7][sh + 0][sw + 1] * filter2[fil][7][0][1];
                            accd081 =   accc081 + accc082;
                            accc083 = ifmaps2[7][sh + 0][sw + 2] * filter2[fil][7][0][2];
                            accc084 = ifmaps2[7][sh + 1][sw + 0] * filter2[fil][7][1][0];
                            accd082 =   accc083 + accc084;
                            acce081 =       accd081 + accd082;
                            accc085 = ifmaps2[7][sh + 1][sw + 1] * filter2[fil][7][1][1];
                            accc086 = ifmaps2[7][sh + 1][sw + 2] * filter2[fil][7][1][2];
                            accd083 =   accc085 + accc086;
                            accc087 = ifmaps2[7][sh + 2][sw + 0] * filter2[fil][7][2][0];
                            accc088 = ifmaps2[7][sh + 2][sw + 1] * filter2[fil][7][2][1];
                            accd084 =   accc087 + accc088;
                            acce082 =       accd083 + accd084;
                            accc089 = ifmaps2[7][sh + 2][sw + 2] * filter2[fil][7][2][2];
                            acc8 =  acce081 + acce082 + accc089;

                            ac4 = acc7 + acc8;
                            a2 = ac3 + ac4;
                            a_c1 = a1 + a2;

                            accc091 = ifmaps2[8][sh + 0][sw + 0] * filter2[fil][8][0][0];
                            accc092 = ifmaps2[8][sh + 0][sw + 1] * filter2[fil][8][0][1];
                            accd091 =   accc091 + accc092;
                            accc093 = ifmaps2[8][sh + 0][sw + 2] * filter2[fil][8][0][2];
                            accc094 = ifmaps2[8][sh + 1][sw + 0] * filter2[fil][8][1][0];
                            accd092 =   accc093 + accc094;
                            acce091 =       accd091 + accd092;
                            accc095 = ifmaps2[8][sh + 1][sw + 1] * filter2[fil][8][1][1];
                            accc096 = ifmaps2[8][sh + 1][sw + 2] * filter2[fil][8][1][2];
                            accd093 =   accc095 + accc096;
                            accc097 = ifmaps2[8][sh + 2][sw + 0] * filter2[fil][8][2][0];
                            accc098 = ifmaps2[8][sh + 2][sw + 1] * filter2[fil][8][2][1];
                            accd094 =   accc097 + accc098;
                            acce092 =       accd093 + accd094;
                            accc099 = ifmaps2[8][sh + 2][sw + 2] * filter2[fil][8][2][2];
                            acc9 =  acce091 + acce092 + accc099;

                            accc101 = ifmaps2[9][sh + 0][sw + 0] * filter2[fil][9][0][0];
                            accc102 = ifmaps2[9][sh + 0][sw + 1] * filter2[fil][9][0][1];
                            accd101 =   accc101 + accc102;
                            accc103 = ifmaps2[9][sh + 0][sw + 2] * filter2[fil][9][0][2];
                            accc104 = ifmaps2[9][sh + 1][sw + 0] * filter2[fil][9][1][0];
                            accd102 =   accc103 + accc104;
                            acce101 =       accd101 + accd102;
                            accc105 = ifmaps2[9][sh + 1][sw + 1] * filter2[fil][9][1][1];
                            accc106 = ifmaps2[9][sh + 1][sw + 2] * filter2[fil][9][1][2];
                            accd103 =   accc105 + accc106;
                            accc107 = ifmaps2[9][sh + 2][sw + 0] * filter2[fil][9][2][0];
                            accc108 = ifmaps2[9][sh + 2][sw + 1] * filter2[fil][9][2][1];
                            accd104 =   accc107 + accc108;
                            acce102 =       accd103 + accd104;
                            accc109 = ifmaps2[9][sh + 2][sw + 2] * filter2[fil][9][2][2];
                            acc10 =  acce101 + acce102 + accc109;

                            ac5 = acc9 + acc10;

                            accc111 = ifmaps2[10][sh + 0][sw + 0] * filter2[fil][10][0][0];
                            accc112 = ifmaps2[10][sh + 0][sw + 1] * filter2[fil][10][0][1];
                            accd111 =   accc111 + accc112;
                            accc113 = ifmaps2[10][sh + 0][sw + 2] * filter2[fil][10][0][2];
                            accc114 = ifmaps2[10][sh + 1][sw + 0] * filter2[fil][10][1][0];
                            accd112 =   accc113 + accc114;
                            acce111 =       accd111 + accd112;
                            accc115 = ifmaps2[10][sh + 1][sw + 1] * filter2[fil][10][1][1];
                            accc116 = ifmaps2[10][sh + 1][sw + 2] * filter2[fil][10][1][2];
                            accd113 =   accc115 + accc116;
                            accc117 = ifmaps2[10][sh + 2][sw + 0] * filter2[fil][10][2][0];
                            accc118 = ifmaps2[10][sh + 2][sw + 1] * filter2[fil][10][2][1];
                            accd114 =   accc117 + accc118;
                            acce112 =       accd113 + accd114;
                            accc119 = ifmaps2[10][sh + 2][sw + 2] * filter2[fil][10][2][2];
                            acc11 =  acce111 + acce112 + accc119;

                            accc121 = ifmaps2[11][sh + 0][sw + 0] * filter2[fil][11][0][0];
                            accc122 = ifmaps2[11][sh + 0][sw + 1] * filter2[fil][11][0][1];
                            accd121 =   accc121 + accc122;
                            accc123 = ifmaps2[11][sh + 0][sw + 2] * filter2[fil][11][0][2];
                            accc124 = ifmaps2[11][sh + 1][sw + 0] * filter2[fil][11][1][0];
                            accd122 =   accc123 + accc124;
                            acce121 =       accd121 + accd122;
                            accc125 = ifmaps2[11][sh + 1][sw + 1] * filter2[fil][11][1][1];
                            accc126 = ifmaps2[11][sh + 1][sw + 2] * filter2[fil][11][1][2];
                            accd123 =   accc125 + accc126;
                            accc127 = ifmaps2[11][sh + 2][sw + 0] * filter2[fil][11][2][0];
                            accc128 = ifmaps2[11][sh + 2][sw + 1] * filter2[fil][11][2][1];
                            accd124 =   accc127 + accc128;
                            acce122 =       accd123 + accd124;
                            accc129 = ifmaps2[11][sh + 2][sw + 2] * filter2[fil][11][2][2];
                            acc12 =  acce121 + acce122 + accc129;

                            ac6 = acc11 + acc12;
                            a3 = ac5 + ac6;

                            accc131 = ifmaps2[12][sh + 0][sw + 0] * filter2[fil][12][0][0];
                            accc132 = ifmaps2[12][sh + 0][sw + 1] * filter2[fil][12][0][1];
                            accd131 =   accc131 + accc132;
                            accc133 = ifmaps2[12][sh + 0][sw + 2] * filter2[fil][12][0][2];
                            accc134 = ifmaps2[12][sh + 1][sw + 0] * filter2[fil][12][1][0];
                            accd132 =   accc133 + accc134;
                            acce131 =       accd131 + accd132;
                            accc135 = ifmaps2[12][sh + 1][sw + 1] * filter2[fil][12][1][1];
                            accc136 = ifmaps2[12][sh + 1][sw + 2] * filter2[fil][12][1][2];
                            accd133 =   accc135 + accc136;
                            accc137 = ifmaps2[12][sh + 2][sw + 0] * filter2[fil][12][2][0];
                            accc138 = ifmaps2[12][sh + 2][sw + 1] * filter2[fil][12][2][1];
                            accd134 =   accc137 + accc138;
                            acce132 =       accd133 + accd134;
                            accc139 = ifmaps2[12][sh + 2][sw + 2] * filter2[fil][12][2][2];
                            acc13 =  acce131 + acce132 + accc139;

                            accc141 = ifmaps2[13][sh + 0][sw + 0] * filter2[fil][13][0][0];
                            accc142 = ifmaps2[13][sh + 0][sw + 1] * filter2[fil][13][0][1];
                            accd141 =   accc141 + accc142;
                            accc143 = ifmaps2[13][sh + 0][sw + 2] * filter2[fil][13][0][2];
                            accc144 = ifmaps2[13][sh + 1][sw + 0] * filter2[fil][13][1][0];
                            accd142 =   accc143 + accc144;
                            acce141 =       accd141 + accd142;
                            accc145 = ifmaps2[13][sh + 1][sw + 1] * filter2[fil][13][1][1];
                            accc146 = ifmaps2[13][sh + 1][sw + 2] * filter2[fil][13][1][2];
                            accd143 =   accc145 + accc146;
                            accc147 = ifmaps2[13][sh + 2][sw + 0] * filter2[fil][13][2][0];
                            accc148 = ifmaps2[13][sh + 2][sw + 1] * filter2[fil][13][2][1];
                            accd144 =   accc147 + accc148;
                            acce142 =       accd143 + accd144;
                            accc149 = ifmaps2[13][sh + 2][sw + 2] * filter2[fil][13][2][2];
                            acc14 =  acce141 + acce142 + accc149;

                            ac7 = acc13 + acc14;

                            accc151 = ifmaps2[14][sh + 0][sw + 0] * filter2[fil][14][0][0];
                            accc152 = ifmaps2[14][sh + 0][sw + 1] * filter2[fil][14][0][1];
                            accd151 =   accc151 + accc152;
                            accc153 = ifmaps2[14][sh + 0][sw + 2] * filter2[fil][14][0][2];
                            accc154 = ifmaps2[14][sh + 1][sw + 0] * filter2[fil][14][1][0];
                            accd152 =   accc153 + accc154;
                            acce151 =       accd151 + accd152;
                            accc155 = ifmaps2[14][sh + 1][sw + 1] * filter2[fil][14][1][1];
                            accc156 = ifmaps2[14][sh + 1][sw + 2] * filter2[fil][14][1][2];
                            accd153 =   accc155 + accc156;
                            accc157 = ifmaps2[14][sh + 2][sw + 0] * filter2[fil][14][2][0];
                            accc158 = ifmaps2[14][sh + 2][sw + 1] * filter2[fil][14][2][1];
                            accd154 =   accc157 + accc158;
                            acce152 =       accd153 + accd154;
                            accc159 = ifmaps2[14][sh + 2][sw + 2] * filter2[fil][14][2][2];
                            acc15 =  acce151 + acce152 + accc159;

                            accc161 = ifmaps2[15][sh + 0][sw + 0] * filter2[fil][15][0][0];
                            accc162 = ifmaps2[15][sh + 0][sw + 1] * filter2[fil][15][0][1];
                            accd161 =   accc161 + accc162;
                            accc163 = ifmaps2[15][sh + 0][sw + 2] * filter2[fil][15][0][2];
                            accc164 = ifmaps2[15][sh + 1][sw + 0] * filter2[fil][15][1][0];
                            accd162 =   accc163 + accc164;
                            acce161 =       accd161 + accd162;
                            accc165 = ifmaps2[15][sh + 1][sw + 1] * filter2[fil][15][1][1];
                            accc166 = ifmaps2[15][sh + 1][sw + 2] * filter2[fil][15][1][2];
                            accd163 =   accc165 + accc166;
                            accc167 = ifmaps2[15][sh + 2][sw + 0] * filter2[fil][15][2][0];
                            accc168 = ifmaps2[15][sh + 2][sw + 1] * filter2[fil][15][2][1];
                            accd164 =   accc167 + accc168;
                            acce162 =       accd163 + accd164;
                            accc169 = ifmaps2[15][sh + 2][sw + 2] * filter2[fil][15][2][2];
                            acc16 =  acce161 + acce162 + accc169;

                            ac8 = acc15 + acc16;
                            a4 = ac7 + ac8;
                            a_c2 = a3 + a4;

                            // a1 + a2 + a3 + a4;   // a_c1 + a_c2;
                            //acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + + acc10  + acc11  + acc12  + acc13  + acc14  + acc15  + acc16;
                            ofmaps[ofm_idx] += a_c1 + a_c2;
                            ofm_idx += dst_h * dst_w;


                        }   // for fil
                    }   // for sw
                }   // for sh
            }   // for chan

            int icp_ofm = 0;
            for(; icp_ofm < dst_c*dst_h*dst_w; icp_ofm++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                C[icp_ofm] = ofmaps[icp_ofm];
            }
            // actor机制
            if(icp_ofm == dst_c*dst_h*dst_w){
                for(;icp_ofm < ofmelmt;){
                    #ifdef SDSOC
                    #pragma HLS PIPELINE
                    #endif
                    C[icp_ofm] = 0;
                    icp_ofm++;
                }
                //std::cout << "actor on, the last icp_ofm = " << icp_ofm - 1 << std::endl;
            }

            break;
        }   

        case CONV4:{
            const int src_c = 128;
            const int src_h = 15;
            const int src_w = 15;
            const int stride = 1;
            const int filter_size = 128*3*3;
            const int filter_h = 3;
            const int filter_w = 3;
            const int dst_c = 256;
            const int dst_h = 13;
            const int dst_w = 13;
            const int end_h = src_h - filter_h + 1;
            const int end_w = src_w - filter_w + 1;

            // 给ofmaps进行初始化
            for(int i = 0; i < dst_c*dst_h*dst_w; i++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                ofmaps[i] = 0;
            }

            for (int icp_ifm = 0, icp_fil = 0,  chan = 0; chan < src_c; chan += fac_para){        // 一个filter的channel维度。其中若chan+=4代表4个filter channel在并行。
                // 用local memory ifmaps3来存储输入。
                for(int cidx = 0; cidx < fac_para; cidx++){
                    for(int hidx = 0; hidx < src_h; hidx++){
                        for(int widx = 0; widx < src_w; widx++){
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            #endif
                            ifmaps3[cidx][hidx][widx] = A[icp_ifm];
                            icp_ifm++;
                        }
                    }
                }
                // actor机制
                if(icp_ifm == src_c*src_h*src_w){
                    for(;icp_ifm < ifmelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif                    
                        actor_1 = A[icp_ifm];
                        icp_ifm++;                              
                    }
                    //std::cout << "actor on, the last icp_ifm = " << icp_ifm - 1 << std::endl;  
                }   
                // 用local memory来存储filter。
                for(int fil = 0; fil < dst_c; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < filter_h; hidx++){
                            for(int widx = 0; widx < filter_w; widx++){
                                #ifdef SDSOC
                                #pragma HLS PIPELINE
                                #endif
                                filter2[fil][cidx][hidx][widx] = B[icp_fil];
                                icp_fil++;
                            }
                        }
                    }
                }
                // actor机制
                if(icp_fil == dst_c*src_c*filter_h*filter_w){
                    for(;icp_fil < filelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif
                        actor_2 = B[icp_fil];
                        icp_fil++;
                    }
                    //std::cout << "actor on, the last icp_fil = " << icp_fil - 1 << std::endl;  
                }  
                // 移窗
                for(int sh = 0; sh < end_h; sh += stride){                  // 纵向移窗。
                    for (int sw = 0; sw < end_w; sw += stride){             // 横向移窗。
                        int ofm_idx = sh/stride * dst_w + sw/stride;
                        for(int fil = 0; fil < dst_c; fil++){      // 切换filter。
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            //#pragma HLS dependence array inter false
                            #endif
                            accc011 = ifmaps3[0][sh + 0][sw + 0] * filter2[fil][0][0][0];
                            accc012 = ifmaps3[0][sh + 0][sw + 1] * filter2[fil][0][0][1];
                            accd011 =   accc011 + accc012;
                            accc013 = ifmaps3[0][sh + 0][sw + 2] * filter2[fil][0][0][2];
                            accc014 = ifmaps3[0][sh + 1][sw + 0] * filter2[fil][0][1][0];
                            accd012 =   accc013 + accc014;
                            acce011 =       accd011 + accd012;
                            accc015 = ifmaps3[0][sh + 1][sw + 1] * filter2[fil][0][1][1];
                            accc016 = ifmaps3[0][sh + 1][sw + 2] * filter2[fil][0][1][2];
                            accd013 =   accc015 + accc016;
                            accc017 = ifmaps3[0][sh + 2][sw + 0] * filter2[fil][0][2][0];
                            accc018 = ifmaps3[0][sh + 2][sw + 1] * filter2[fil][0][2][1];
                            accd014 =   accc017 + accc018;
                            acce012 =       accd013 + accd014;
                            accc019 = ifmaps3[0][sh + 2][sw + 2] * filter2[fil][0][2][2];
                            acc1 =  acce011 + acce012 + accc019;

                            accc021 = ifmaps3[1][sh + 0][sw + 0] * filter2[fil][1][0][0];
                            accc022 = ifmaps3[1][sh + 0][sw + 1] * filter2[fil][1][0][1];
                            accd021 =   accc021 + accc022;
                            accc023 = ifmaps3[1][sh + 0][sw + 2] * filter2[fil][1][0][2];
                            accc024 = ifmaps3[1][sh + 1][sw + 0] * filter2[fil][1][1][0];
                            accd022 =   accc023 + accc024;
                            acce021 =       accd021 + accd022;
                            accc025 = ifmaps3[1][sh + 1][sw + 1] * filter2[fil][1][1][1];
                            accc026 = ifmaps3[1][sh + 1][sw + 2] * filter2[fil][1][1][2];
                            accd023 =   accc025 + accc026;
                            accc027 = ifmaps3[1][sh + 2][sw + 0] * filter2[fil][1][2][0];
                            accc028 = ifmaps3[1][sh + 2][sw + 1] * filter2[fil][1][2][1];
                            accd024 =   accc027 + accc028;
                            acce022 =       accd023 + accd024;
                            accc029 = ifmaps3[1][sh + 2][sw + 2] * filter2[fil][1][2][2];
                            acc2 =  acce021 + acce022 + accc029;

                            ac1 = acc1 + acc2;

                            accc031 = ifmaps3[2][sh + 0][sw + 0] * filter2[fil][2][0][0];
                            accc032 = ifmaps3[2][sh + 0][sw + 1] * filter2[fil][2][0][1];
                            accd031 =   accc031 + accc032;
                            accc033 = ifmaps3[2][sh + 0][sw + 2] * filter2[fil][2][0][2];
                            accc034 = ifmaps3[2][sh + 1][sw + 0] * filter2[fil][2][1][0];
                            accd032 =   accc033 + accc034;
                            acce031 =       accd031 + accd032;
                            accc035 = ifmaps3[2][sh + 1][sw + 1] * filter2[fil][2][1][1];
                            accc036 = ifmaps3[2][sh + 1][sw + 2] * filter2[fil][2][1][2];
                            accd033 =   accc035 + accc036;
                            accc037 = ifmaps3[2][sh + 2][sw + 0] * filter2[fil][2][2][0];
                            accc038 = ifmaps3[2][sh + 2][sw + 1] * filter2[fil][2][2][1];
                            accd034 =   accc037 + accc038;
                            acce032 =       accd033 + accd034;
                            accc039 = ifmaps3[2][sh + 2][sw + 2] * filter2[fil][2][2][2];
                            acc3 =  acce031 + acce032 + accc039;

                            accc041 = ifmaps3[3][sh + 0][sw + 0] * filter2[fil][3][0][0];
                            accc042 = ifmaps3[3][sh + 0][sw + 1] * filter2[fil][3][0][1];
                            accd041 =   accc041 + accc042;
                            accc043 = ifmaps3[3][sh + 0][sw + 2] * filter2[fil][3][0][2];
                            accc044 = ifmaps3[3][sh + 1][sw + 0] * filter2[fil][3][1][0];
                            accd042 =   accc043 + accc044;
                            acce041 =       accd041 + accd042;
                            accc045 = ifmaps3[3][sh + 1][sw + 1] * filter2[fil][3][1][1];
                            accc046 = ifmaps3[3][sh + 1][sw + 2] * filter2[fil][3][1][2];
                            accd043 =   accc045 + accc046;
                            accc047 = ifmaps3[3][sh + 2][sw + 0] * filter2[fil][3][2][0];
                            accc048 = ifmaps3[3][sh + 2][sw + 1] * filter2[fil][3][2][1];
                            accd044 =   accc047 + accc048;
                            acce042 =       accd043 + accd044;
                            accc049 = ifmaps3[3][sh + 2][sw + 2] * filter2[fil][3][2][2];
                            acc4 =  acce041 + acce042 + accc049;

                            ac2 = acc3 +acc4;
                            a1 = ac1 + ac2;

                            accc051 = ifmaps3[4][sh + 0][sw + 0] * filter2[fil][4][0][0];
                            accc052 = ifmaps3[4][sh + 0][sw + 1] * filter2[fil][4][0][1];
                            accd051 =   accc051 + accc052;
                            accc053 = ifmaps3[4][sh + 0][sw + 2] * filter2[fil][4][0][2];
                            accc054 = ifmaps3[4][sh + 1][sw + 0] * filter2[fil][4][1][0];
                            accd052 =   accc053 + accc054;
                            acce051 =       accd051 + accd052;
                            accc055 = ifmaps3[4][sh + 1][sw + 1] * filter2[fil][4][1][1];
                            accc056 = ifmaps3[4][sh + 1][sw + 2] * filter2[fil][4][1][2];
                            accd053 =   accc055 + accc056;
                            accc057 = ifmaps3[4][sh + 2][sw + 0] * filter2[fil][4][2][0];
                            accc058 = ifmaps3[4][sh + 2][sw + 1] * filter2[fil][4][2][1];
                            accd054 =   accc057 + accc058;
                            acce052 =       accd053 + accd054;
                            accc059 = ifmaps3[4][sh + 2][sw + 2] * filter2[fil][4][2][2];
                            acc5 =  acce051 + acce052 + accc059;

                            accc061 = ifmaps3[5][sh + 0][sw + 0] * filter2[fil][5][0][0];
                            accc062 = ifmaps3[5][sh + 0][sw + 1] * filter2[fil][5][0][1];
                            accd061 =   accc061 + accc062;
                            accc063 = ifmaps3[5][sh + 0][sw + 2] * filter2[fil][5][0][2];
                            accc064 = ifmaps3[5][sh + 1][sw + 0] * filter2[fil][5][1][0];
                            accd062 =   accc063 + accc064;
                            acce061 =       accd061 + accd062;
                            accc065 = ifmaps3[5][sh + 1][sw + 1] * filter2[fil][5][1][1];
                            accc066 = ifmaps3[5][sh + 1][sw + 2] * filter2[fil][5][1][2];
                            accd063 =   accc065 + accc066;
                            accc067 = ifmaps3[5][sh + 2][sw + 0] * filter2[fil][5][2][0];
                            accc068 = ifmaps3[5][sh + 2][sw + 1] * filter2[fil][5][2][1];
                            accd064 =   accc067 + accc068;
                            acce062 =       accd063 + accd064;
                            accc069 = ifmaps3[5][sh + 2][sw + 2] * filter2[fil][5][2][2];
                            acc6 =  acce061 + acce062 + accc069;

                            ac3 = acc5 + acc6;

                            accc071 = ifmaps3[6][sh + 0][sw + 0] * filter2[fil][6][0][0];
                            accc072 = ifmaps3[6][sh + 0][sw + 1] * filter2[fil][6][0][1];
                            accd071 =   accc071 + accc072;
                            accc073 = ifmaps3[6][sh + 0][sw + 2] * filter2[fil][6][0][2];
                            accc074 = ifmaps3[6][sh + 1][sw + 0] * filter2[fil][6][1][0];
                            accd072 =   accc073 + accc074;
                            acce071 =       accd071 + accd072;
                            accc075 = ifmaps3[6][sh + 1][sw + 1] * filter2[fil][6][1][1];
                            accc076 = ifmaps3[6][sh + 1][sw + 2] * filter2[fil][6][1][2];
                            accd073 =   accc075 + accc076;
                            accc077 = ifmaps3[6][sh + 2][sw + 0] * filter2[fil][6][2][0];
                            accc078 = ifmaps3[6][sh + 2][sw + 1] * filter2[fil][6][2][1];
                            accd074 =   accc077 + accc078;
                            acce072 =       accd073 + accd074;
                            accc079 = ifmaps3[6][sh + 2][sw + 2] * filter2[fil][6][2][2];
                            acc7 =  acce071 + acce072 + accc079;

                            accc081 = ifmaps3[7][sh + 0][sw + 0] * filter2[fil][7][0][0];
                            accc082 = ifmaps3[7][sh + 0][sw + 1] * filter2[fil][7][0][1];
                            accd081 =   accc081 + accc082;
                            accc083 = ifmaps3[7][sh + 0][sw + 2] * filter2[fil][7][0][2];
                            accc084 = ifmaps3[7][sh + 1][sw + 0] * filter2[fil][7][1][0];
                            accd082 =   accc083 + accc084;
                            acce081 =       accd081 + accd082;
                            accc085 = ifmaps3[7][sh + 1][sw + 1] * filter2[fil][7][1][1];
                            accc086 = ifmaps3[7][sh + 1][sw + 2] * filter2[fil][7][1][2];
                            accd083 =   accc085 + accc086;
                            accc087 = ifmaps3[7][sh + 2][sw + 0] * filter2[fil][7][2][0];
                            accc088 = ifmaps3[7][sh + 2][sw + 1] * filter2[fil][7][2][1];
                            accd084 =   accc087 + accc088;
                            acce082 =       accd083 + accd084;
                            accc089 = ifmaps3[7][sh + 2][sw + 2] * filter2[fil][7][2][2];
                            acc8 =  acce081 + acce082 + accc089;

                            ac4 = acc7 + acc8;
                            a2 = ac3 + ac4;
                            a_c1 = a1 + a2;

                            accc091 = ifmaps3[8][sh + 0][sw + 0] * filter2[fil][8][0][0];
                            accc092 = ifmaps3[8][sh + 0][sw + 1] * filter2[fil][8][0][1];
                            accd091 =   accc091 + accc092;
                            accc093 = ifmaps3[8][sh + 0][sw + 2] * filter2[fil][8][0][2];
                            accc094 = ifmaps3[8][sh + 1][sw + 0] * filter2[fil][8][1][0];
                            accd092 =   accc093 + accc094;
                            acce091 =       accd091 + accd092;
                            accc095 = ifmaps3[8][sh + 1][sw + 1] * filter2[fil][8][1][1];
                            accc096 = ifmaps3[8][sh + 1][sw + 2] * filter2[fil][8][1][2];
                            accd093 =   accc095 + accc096;
                            accc097 = ifmaps3[8][sh + 2][sw + 0] * filter2[fil][8][2][0];
                            accc098 = ifmaps3[8][sh + 2][sw + 1] * filter2[fil][8][2][1];
                            accd094 =   accc097 + accc098;
                            acce092 =       accd093 + accd094;
                            accc099 = ifmaps3[8][sh + 2][sw + 2] * filter2[fil][8][2][2];
                            acc9 =  acce091 + acce092 + accc099;

                            accc101 = ifmaps3[9][sh + 0][sw + 0] * filter2[fil][9][0][0];
                            accc102 = ifmaps3[9][sh + 0][sw + 1] * filter2[fil][9][0][1];
                            accd101 =   accc101 + accc102;
                            accc103 = ifmaps3[9][sh + 0][sw + 2] * filter2[fil][9][0][2];
                            accc104 = ifmaps3[9][sh + 1][sw + 0] * filter2[fil][9][1][0];
                            accd102 =   accc103 + accc104;
                            acce101 =       accd101 + accd102;
                            accc105 = ifmaps3[9][sh + 1][sw + 1] * filter2[fil][9][1][1];
                            accc106 = ifmaps3[9][sh + 1][sw + 2] * filter2[fil][9][1][2];
                            accd103 =   accc105 + accc106;
                            accc107 = ifmaps3[9][sh + 2][sw + 0] * filter2[fil][9][2][0];
                            accc108 = ifmaps3[9][sh + 2][sw + 1] * filter2[fil][9][2][1];
                            accd104 =   accc107 + accc108;
                            acce102 =       accd103 + accd104;
                            accc109 = ifmaps3[9][sh + 2][sw + 2] * filter2[fil][9][2][2];
                            acc10 =  acce101 + acce102 + accc109;

                            ac5 = acc9 + acc10;

                            accc111 = ifmaps3[10][sh + 0][sw + 0] * filter2[fil][10][0][0];
                            accc112 = ifmaps3[10][sh + 0][sw + 1] * filter2[fil][10][0][1];
                            accd111 =   accc111 + accc112;
                            accc113 = ifmaps3[10][sh + 0][sw + 2] * filter2[fil][10][0][2];
                            accc114 = ifmaps3[10][sh + 1][sw + 0] * filter2[fil][10][1][0];
                            accd112 =   accc113 + accc114;
                            acce111 =       accd111 + accd112;
                            accc115 = ifmaps3[10][sh + 1][sw + 1] * filter2[fil][10][1][1];
                            accc116 = ifmaps3[10][sh + 1][sw + 2] * filter2[fil][10][1][2];
                            accd113 =   accc115 + accc116;
                            accc117 = ifmaps3[10][sh + 2][sw + 0] * filter2[fil][10][2][0];
                            accc118 = ifmaps3[10][sh + 2][sw + 1] * filter2[fil][10][2][1];
                            accd114 =   accc117 + accc118;
                            acce112 =       accd113 + accd114;
                            accc119 = ifmaps3[10][sh + 2][sw + 2] * filter2[fil][10][2][2];
                            acc11 =  acce111 + acce112 + accc119;

                            accc121 = ifmaps3[11][sh + 0][sw + 0] * filter2[fil][11][0][0];
                            accc122 = ifmaps3[11][sh + 0][sw + 1] * filter2[fil][11][0][1];
                            accd121 =   accc121 + accc122;
                            accc123 = ifmaps3[11][sh + 0][sw + 2] * filter2[fil][11][0][2];
                            accc124 = ifmaps3[11][sh + 1][sw + 0] * filter2[fil][11][1][0];
                            accd122 =   accc123 + accc124;
                            acce121 =       accd121 + accd122;
                            accc125 = ifmaps3[11][sh + 1][sw + 1] * filter2[fil][11][1][1];
                            accc126 = ifmaps3[11][sh + 1][sw + 2] * filter2[fil][11][1][2];
                            accd123 =   accc125 + accc126;
                            accc127 = ifmaps3[11][sh + 2][sw + 0] * filter2[fil][11][2][0];
                            accc128 = ifmaps3[11][sh + 2][sw + 1] * filter2[fil][11][2][1];
                            accd124 =   accc127 + accc128;
                            acce122 =       accd123 + accd124;
                            accc129 = ifmaps3[11][sh + 2][sw + 2] * filter2[fil][11][2][2];
                            acc12 =  acce121 + acce122 + accc129;

                            ac6 = acc11 + acc12;
                            a3 = ac5 + ac6;

                            accc131 = ifmaps3[12][sh + 0][sw + 0] * filter2[fil][12][0][0];
                            accc132 = ifmaps3[12][sh + 0][sw + 1] * filter2[fil][12][0][1];
                            accd131 =   accc131 + accc132;
                            accc133 = ifmaps3[12][sh + 0][sw + 2] * filter2[fil][12][0][2];
                            accc134 = ifmaps3[12][sh + 1][sw + 0] * filter2[fil][12][1][0];
                            accd132 =   accc133 + accc134;
                            acce131 =       accd131 + accd132;
                            accc135 = ifmaps3[12][sh + 1][sw + 1] * filter2[fil][12][1][1];
                            accc136 = ifmaps3[12][sh + 1][sw + 2] * filter2[fil][12][1][2];
                            accd133 =   accc135 + accc136;
                            accc137 = ifmaps3[12][sh + 2][sw + 0] * filter2[fil][12][2][0];
                            accc138 = ifmaps3[12][sh + 2][sw + 1] * filter2[fil][12][2][1];
                            accd134 =   accc137 + accc138;
                            acce132 =       accd133 + accd134;
                            accc139 = ifmaps3[12][sh + 2][sw + 2] * filter2[fil][12][2][2];
                            acc13 =  acce131 + acce132 + accc139;

                            accc141 = ifmaps3[13][sh + 0][sw + 0] * filter2[fil][13][0][0];
                            accc142 = ifmaps3[13][sh + 0][sw + 1] * filter2[fil][13][0][1];
                            accd141 =   accc141 + accc142;
                            accc143 = ifmaps3[13][sh + 0][sw + 2] * filter2[fil][13][0][2];
                            accc144 = ifmaps3[13][sh + 1][sw + 0] * filter2[fil][13][1][0];
                            accd142 =   accc143 + accc144;
                            acce141 =       accd141 + accd142;
                            accc145 = ifmaps3[13][sh + 1][sw + 1] * filter2[fil][13][1][1];
                            accc146 = ifmaps3[13][sh + 1][sw + 2] * filter2[fil][13][1][2];
                            accd143 =   accc145 + accc146;
                            accc147 = ifmaps3[13][sh + 2][sw + 0] * filter2[fil][13][2][0];
                            accc148 = ifmaps3[13][sh + 2][sw + 1] * filter2[fil][13][2][1];
                            accd144 =   accc147 + accc148;
                            acce142 =       accd143 + accd144;
                            accc149 = ifmaps3[13][sh + 2][sw + 2] * filter2[fil][13][2][2];
                            acc14 =  acce141 + acce142 + accc149;

                            ac7 = acc13 + acc14;

                            accc151 = ifmaps3[14][sh + 0][sw + 0] * filter2[fil][14][0][0];
                            accc152 = ifmaps3[14][sh + 0][sw + 1] * filter2[fil][14][0][1];
                            accd151 =   accc151 + accc152;
                            accc153 = ifmaps3[14][sh + 0][sw + 2] * filter2[fil][14][0][2];
                            accc154 = ifmaps3[14][sh + 1][sw + 0] * filter2[fil][14][1][0];
                            accd152 =   accc153 + accc154;
                            acce151 =       accd151 + accd152;
                            accc155 = ifmaps3[14][sh + 1][sw + 1] * filter2[fil][14][1][1];
                            accc156 = ifmaps3[14][sh + 1][sw + 2] * filter2[fil][14][1][2];
                            accd153 =   accc155 + accc156;
                            accc157 = ifmaps3[14][sh + 2][sw + 0] * filter2[fil][14][2][0];
                            accc158 = ifmaps3[14][sh + 2][sw + 1] * filter2[fil][14][2][1];
                            accd154 =   accc157 + accc158;
                            acce152 =       accd153 + accd154;
                            accc159 = ifmaps3[14][sh + 2][sw + 2] * filter2[fil][14][2][2];
                            acc15 =  acce151 + acce152 + accc159;

                            accc161 = ifmaps3[15][sh + 0][sw + 0] * filter2[fil][15][0][0];
                            accc162 = ifmaps3[15][sh + 0][sw + 1] * filter2[fil][15][0][1];
                            accd161 =   accc161 + accc162;
                            accc163 = ifmaps3[15][sh + 0][sw + 2] * filter2[fil][15][0][2];
                            accc164 = ifmaps3[15][sh + 1][sw + 0] * filter2[fil][15][1][0];
                            accd162 =   accc163 + accc164;
                            acce161 =       accd161 + accd162;
                            accc165 = ifmaps3[15][sh + 1][sw + 1] * filter2[fil][15][1][1];
                            accc166 = ifmaps3[15][sh + 1][sw + 2] * filter2[fil][15][1][2];
                            accd163 =   accc165 + accc166;
                            accc167 = ifmaps3[15][sh + 2][sw + 0] * filter2[fil][15][2][0];
                            accc168 = ifmaps3[15][sh + 2][sw + 1] * filter2[fil][15][2][1];
                            accd164 =   accc167 + accc168;
                            acce162 =       accd163 + accd164;
                            accc169 = ifmaps3[15][sh + 2][sw + 2] * filter2[fil][15][2][2];
                            acc16 =  acce161 + acce162 + accc169;

                            ac8 = acc15 + acc16;
                            a4 = ac7 + ac8;
                            a_c2 = a3 + a4;

                            // a1 + a2 + a3 + a4;   // a_c1 + a_c2;
                            //acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + + acc10  + acc11  + acc12  + acc13  + acc14  + acc15  + acc16;
                            ofmaps[ofm_idx] += a_c1 + a_c2;
                            ofm_idx += dst_h * dst_w;


                        }   // for fil
                    }   // for sw
                }   // for sh
            }   // for chan

            int icp_ofm = 0;
            for(; icp_ofm < dst_c*dst_h*dst_w; icp_ofm++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                C[icp_ofm] = ofmaps[icp_ofm];
            }
            // actor机制
            if(icp_ofm == dst_c*dst_h*dst_w){
                for(;icp_ofm < ofmelmt;){
                    #ifdef SDSOC
                    #pragma HLS PIPELINE
                    #endif
                    C[icp_ofm] = 0;
                    icp_ofm++;
                }
                //std::cout << "actor on, the last icp_ofm = " << icp_ofm - 1 << std::endl;
            }

            break;
        }

        case CONV5:{
            const int src_c = 256;
            const int src_h = 15;
            const int src_w = 15;
            const int stride = 1;
            const int filter_size = 256*3*3;
            const int filter_h = 3;
            const int filter_w = 3;
            const int dst_c = 192;
            const int dst_h = 13;
            const int dst_w = 13;
            const int end_h = src_h - filter_h + 1;
            const int end_w = src_w - filter_w + 1;

            // 给ofmaps进行初始化
            for(int i = 0; i < dst_c*dst_h*dst_w; i++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                ofmaps[i] = 0;
            }

            for (int icp_ifm = 0, icp_fil = 0,  chan = 0; chan < src_c; chan += fac_para){        // 一个filter的channel维度。其中若chan+=4代表4个filter channel在并行。
                // 用local memory ifmaps3来存储输入。
                for(int cidx = 0; cidx < fac_para; cidx++){
                    for(int hidx = 0; hidx < src_h; hidx++){
                        for(int widx = 0; widx < src_w; widx++){
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            #endif
                            ifmaps3[cidx][hidx][widx] = A[icp_ifm];
                            icp_ifm++;
                        }
                    }
                }
                // actor机制
                if(icp_ifm == src_c*src_h*src_w){
                    for(;icp_ifm < ifmelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif                    
                        actor_1 = A[icp_ifm];
                        icp_ifm++;                              
                    }
                    //std::cout << "actor on, the last icp_ifm = " << icp_ifm - 1 << std::endl;  
                }   
                // 用local memory来存储filter。
                for(int fil = 0; fil < dst_c; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < filter_h; hidx++){
                            for(int widx = 0; widx < filter_w; widx++){
                                #ifdef SDSOC
                                #pragma HLS PIPELINE
                                #endif
                                filter2[fil][cidx][hidx][widx] = B[icp_fil];
                                icp_fil++;
                            }
                        }
                    }
                }
                // actor机制
                if(icp_fil == dst_c*src_c*filter_h*filter_w){
                    for(;icp_fil < filelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif
                        actor_2 = B[icp_fil];
                        icp_fil++;
                    }
                    //std::cout << "actor on, the last icp_fil = " << icp_fil - 1 << std::endl;  
                }  
                // 移窗
                for(int sh = 0; sh < end_h; sh += stride){                  // 纵向移窗。
                    for (int sw = 0; sw < end_w; sw += stride){             // 横向移窗。
                        int ofm_idx = sh/stride * dst_w + sw/stride;
                        for(int fil = 0; fil < dst_c; fil++){      // 切换filter。
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            //#pragma HLS dependence array inter false
                            #endif
                            accc011 = ifmaps3[0][sh + 0][sw + 0] * filter2[fil][0][0][0];
                            accc012 = ifmaps3[0][sh + 0][sw + 1] * filter2[fil][0][0][1];
                            accd011 =   accc011 + accc012;
                            accc013 = ifmaps3[0][sh + 0][sw + 2] * filter2[fil][0][0][2];
                            accc014 = ifmaps3[0][sh + 1][sw + 0] * filter2[fil][0][1][0];
                            accd012 =   accc013 + accc014;
                            acce011 =       accd011 + accd012;
                            accc015 = ifmaps3[0][sh + 1][sw + 1] * filter2[fil][0][1][1];
                            accc016 = ifmaps3[0][sh + 1][sw + 2] * filter2[fil][0][1][2];
                            accd013 =   accc015 + accc016;
                            accc017 = ifmaps3[0][sh + 2][sw + 0] * filter2[fil][0][2][0];
                            accc018 = ifmaps3[0][sh + 2][sw + 1] * filter2[fil][0][2][1];
                            accd014 =   accc017 + accc018;
                            acce012 =       accd013 + accd014;
                            accc019 = ifmaps3[0][sh + 2][sw + 2] * filter2[fil][0][2][2];
                            acc1 =  acce011 + acce012 + accc019;

                            accc021 = ifmaps3[1][sh + 0][sw + 0] * filter2[fil][1][0][0];
                            accc022 = ifmaps3[1][sh + 0][sw + 1] * filter2[fil][1][0][1];
                            accd021 =   accc021 + accc022;
                            accc023 = ifmaps3[1][sh + 0][sw + 2] * filter2[fil][1][0][2];
                            accc024 = ifmaps3[1][sh + 1][sw + 0] * filter2[fil][1][1][0];
                            accd022 =   accc023 + accc024;
                            acce021 =       accd021 + accd022;
                            accc025 = ifmaps3[1][sh + 1][sw + 1] * filter2[fil][1][1][1];
                            accc026 = ifmaps3[1][sh + 1][sw + 2] * filter2[fil][1][1][2];
                            accd023 =   accc025 + accc026;
                            accc027 = ifmaps3[1][sh + 2][sw + 0] * filter2[fil][1][2][0];
                            accc028 = ifmaps3[1][sh + 2][sw + 1] * filter2[fil][1][2][1];
                            accd024 =   accc027 + accc028;
                            acce022 =       accd023 + accd024;
                            accc029 = ifmaps3[1][sh + 2][sw + 2] * filter2[fil][1][2][2];
                            acc2 =  acce021 + acce022 + accc029;

                            ac1 = acc1 + acc2;

                            accc031 = ifmaps3[2][sh + 0][sw + 0] * filter2[fil][2][0][0];
                            accc032 = ifmaps3[2][sh + 0][sw + 1] * filter2[fil][2][0][1];
                            accd031 =   accc031 + accc032;
                            accc033 = ifmaps3[2][sh + 0][sw + 2] * filter2[fil][2][0][2];
                            accc034 = ifmaps3[2][sh + 1][sw + 0] * filter2[fil][2][1][0];
                            accd032 =   accc033 + accc034;
                            acce031 =       accd031 + accd032;
                            accc035 = ifmaps3[2][sh + 1][sw + 1] * filter2[fil][2][1][1];
                            accc036 = ifmaps3[2][sh + 1][sw + 2] * filter2[fil][2][1][2];
                            accd033 =   accc035 + accc036;
                            accc037 = ifmaps3[2][sh + 2][sw + 0] * filter2[fil][2][2][0];
                            accc038 = ifmaps3[2][sh + 2][sw + 1] * filter2[fil][2][2][1];
                            accd034 =   accc037 + accc038;
                            acce032 =       accd033 + accd034;
                            accc039 = ifmaps3[2][sh + 2][sw + 2] * filter2[fil][2][2][2];
                            acc3 =  acce031 + acce032 + accc039;

                            accc041 = ifmaps3[3][sh + 0][sw + 0] * filter2[fil][3][0][0];
                            accc042 = ifmaps3[3][sh + 0][sw + 1] * filter2[fil][3][0][1];
                            accd041 =   accc041 + accc042;
                            accc043 = ifmaps3[3][sh + 0][sw + 2] * filter2[fil][3][0][2];
                            accc044 = ifmaps3[3][sh + 1][sw + 0] * filter2[fil][3][1][0];
                            accd042 =   accc043 + accc044;
                            acce041 =       accd041 + accd042;
                            accc045 = ifmaps3[3][sh + 1][sw + 1] * filter2[fil][3][1][1];
                            accc046 = ifmaps3[3][sh + 1][sw + 2] * filter2[fil][3][1][2];
                            accd043 =   accc045 + accc046;
                            accc047 = ifmaps3[3][sh + 2][sw + 0] * filter2[fil][3][2][0];
                            accc048 = ifmaps3[3][sh + 2][sw + 1] * filter2[fil][3][2][1];
                            accd044 =   accc047 + accc048;
                            acce042 =       accd043 + accd044;
                            accc049 = ifmaps3[3][sh + 2][sw + 2] * filter2[fil][3][2][2];
                            acc4 =  acce041 + acce042 + accc049;

                            ac2 = acc3 +acc4;
                            a1 = ac1 + ac2;

                            accc051 = ifmaps3[4][sh + 0][sw + 0] * filter2[fil][4][0][0];
                            accc052 = ifmaps3[4][sh + 0][sw + 1] * filter2[fil][4][0][1];
                            accd051 =   accc051 + accc052;
                            accc053 = ifmaps3[4][sh + 0][sw + 2] * filter2[fil][4][0][2];
                            accc054 = ifmaps3[4][sh + 1][sw + 0] * filter2[fil][4][1][0];
                            accd052 =   accc053 + accc054;
                            acce051 =       accd051 + accd052;
                            accc055 = ifmaps3[4][sh + 1][sw + 1] * filter2[fil][4][1][1];
                            accc056 = ifmaps3[4][sh + 1][sw + 2] * filter2[fil][4][1][2];
                            accd053 =   accc055 + accc056;
                            accc057 = ifmaps3[4][sh + 2][sw + 0] * filter2[fil][4][2][0];
                            accc058 = ifmaps3[4][sh + 2][sw + 1] * filter2[fil][4][2][1];
                            accd054 =   accc057 + accc058;
                            acce052 =       accd053 + accd054;
                            accc059 = ifmaps3[4][sh + 2][sw + 2] * filter2[fil][4][2][2];
                            acc5 =  acce051 + acce052 + accc059;

                            accc061 = ifmaps3[5][sh + 0][sw + 0] * filter2[fil][5][0][0];
                            accc062 = ifmaps3[5][sh + 0][sw + 1] * filter2[fil][5][0][1];
                            accd061 =   accc061 + accc062;
                            accc063 = ifmaps3[5][sh + 0][sw + 2] * filter2[fil][5][0][2];
                            accc064 = ifmaps3[5][sh + 1][sw + 0] * filter2[fil][5][1][0];
                            accd062 =   accc063 + accc064;
                            acce061 =       accd061 + accd062;
                            accc065 = ifmaps3[5][sh + 1][sw + 1] * filter2[fil][5][1][1];
                            accc066 = ifmaps3[5][sh + 1][sw + 2] * filter2[fil][5][1][2];
                            accd063 =   accc065 + accc066;
                            accc067 = ifmaps3[5][sh + 2][sw + 0] * filter2[fil][5][2][0];
                            accc068 = ifmaps3[5][sh + 2][sw + 1] * filter2[fil][5][2][1];
                            accd064 =   accc067 + accc068;
                            acce062 =       accd063 + accd064;
                            accc069 = ifmaps3[5][sh + 2][sw + 2] * filter2[fil][5][2][2];
                            acc6 =  acce061 + acce062 + accc069;

                            ac3 = acc5 + acc6;

                            accc071 = ifmaps3[6][sh + 0][sw + 0] * filter2[fil][6][0][0];
                            accc072 = ifmaps3[6][sh + 0][sw + 1] * filter2[fil][6][0][1];
                            accd071 =   accc071 + accc072;
                            accc073 = ifmaps3[6][sh + 0][sw + 2] * filter2[fil][6][0][2];
                            accc074 = ifmaps3[6][sh + 1][sw + 0] * filter2[fil][6][1][0];
                            accd072 =   accc073 + accc074;
                            acce071 =       accd071 + accd072;
                            accc075 = ifmaps3[6][sh + 1][sw + 1] * filter2[fil][6][1][1];
                            accc076 = ifmaps3[6][sh + 1][sw + 2] * filter2[fil][6][1][2];
                            accd073 =   accc075 + accc076;
                            accc077 = ifmaps3[6][sh + 2][sw + 0] * filter2[fil][6][2][0];
                            accc078 = ifmaps3[6][sh + 2][sw + 1] * filter2[fil][6][2][1];
                            accd074 =   accc077 + accc078;
                            acce072 =       accd073 + accd074;
                            accc079 = ifmaps3[6][sh + 2][sw + 2] * filter2[fil][6][2][2];
                            acc7 =  acce071 + acce072 + accc079;

                            accc081 = ifmaps3[7][sh + 0][sw + 0] * filter2[fil][7][0][0];
                            accc082 = ifmaps3[7][sh + 0][sw + 1] * filter2[fil][7][0][1];
                            accd081 =   accc081 + accc082;
                            accc083 = ifmaps3[7][sh + 0][sw + 2] * filter2[fil][7][0][2];
                            accc084 = ifmaps3[7][sh + 1][sw + 0] * filter2[fil][7][1][0];
                            accd082 =   accc083 + accc084;
                            acce081 =       accd081 + accd082;
                            accc085 = ifmaps3[7][sh + 1][sw + 1] * filter2[fil][7][1][1];
                            accc086 = ifmaps3[7][sh + 1][sw + 2] * filter2[fil][7][1][2];
                            accd083 =   accc085 + accc086;
                            accc087 = ifmaps3[7][sh + 2][sw + 0] * filter2[fil][7][2][0];
                            accc088 = ifmaps3[7][sh + 2][sw + 1] * filter2[fil][7][2][1];
                            accd084 =   accc087 + accc088;
                            acce082 =       accd083 + accd084;
                            accc089 = ifmaps3[7][sh + 2][sw + 2] * filter2[fil][7][2][2];
                            acc8 =  acce081 + acce082 + accc089;

                            ac4 = acc7 + acc8;
                            a2 = ac3 + ac4;
                            a_c1 = a1 + a2;

                            accc091 = ifmaps3[8][sh + 0][sw + 0] * filter2[fil][8][0][0];
                            accc092 = ifmaps3[8][sh + 0][sw + 1] * filter2[fil][8][0][1];
                            accd091 =   accc091 + accc092;
                            accc093 = ifmaps3[8][sh + 0][sw + 2] * filter2[fil][8][0][2];
                            accc094 = ifmaps3[8][sh + 1][sw + 0] * filter2[fil][8][1][0];
                            accd092 =   accc093 + accc094;
                            acce091 =       accd091 + accd092;
                            accc095 = ifmaps3[8][sh + 1][sw + 1] * filter2[fil][8][1][1];
                            accc096 = ifmaps3[8][sh + 1][sw + 2] * filter2[fil][8][1][2];
                            accd093 =   accc095 + accc096;
                            accc097 = ifmaps3[8][sh + 2][sw + 0] * filter2[fil][8][2][0];
                            accc098 = ifmaps3[8][sh + 2][sw + 1] * filter2[fil][8][2][1];
                            accd094 =   accc097 + accc098;
                            acce092 =       accd093 + accd094;
                            accc099 = ifmaps3[8][sh + 2][sw + 2] * filter2[fil][8][2][2];
                            acc9 =  acce091 + acce092 + accc099;

                            accc101 = ifmaps3[9][sh + 0][sw + 0] * filter2[fil][9][0][0];
                            accc102 = ifmaps3[9][sh + 0][sw + 1] * filter2[fil][9][0][1];
                            accd101 =   accc101 + accc102;
                            accc103 = ifmaps3[9][sh + 0][sw + 2] * filter2[fil][9][0][2];
                            accc104 = ifmaps3[9][sh + 1][sw + 0] * filter2[fil][9][1][0];
                            accd102 =   accc103 + accc104;
                            acce101 =       accd101 + accd102;
                            accc105 = ifmaps3[9][sh + 1][sw + 1] * filter2[fil][9][1][1];
                            accc106 = ifmaps3[9][sh + 1][sw + 2] * filter2[fil][9][1][2];
                            accd103 =   accc105 + accc106;
                            accc107 = ifmaps3[9][sh + 2][sw + 0] * filter2[fil][9][2][0];
                            accc108 = ifmaps3[9][sh + 2][sw + 1] * filter2[fil][9][2][1];
                            accd104 =   accc107 + accc108;
                            acce102 =       accd103 + accd104;
                            accc109 = ifmaps3[9][sh + 2][sw + 2] * filter2[fil][9][2][2];
                            acc10 =  acce101 + acce102 + accc109;

                            ac5 = acc9 + acc10;

                            accc111 = ifmaps3[10][sh + 0][sw + 0] * filter2[fil][10][0][0];
                            accc112 = ifmaps3[10][sh + 0][sw + 1] * filter2[fil][10][0][1];
                            accd111 =   accc111 + accc112;
                            accc113 = ifmaps3[10][sh + 0][sw + 2] * filter2[fil][10][0][2];
                            accc114 = ifmaps3[10][sh + 1][sw + 0] * filter2[fil][10][1][0];
                            accd112 =   accc113 + accc114;
                            acce111 =       accd111 + accd112;
                            accc115 = ifmaps3[10][sh + 1][sw + 1] * filter2[fil][10][1][1];
                            accc116 = ifmaps3[10][sh + 1][sw + 2] * filter2[fil][10][1][2];
                            accd113 =   accc115 + accc116;
                            accc117 = ifmaps3[10][sh + 2][sw + 0] * filter2[fil][10][2][0];
                            accc118 = ifmaps3[10][sh + 2][sw + 1] * filter2[fil][10][2][1];
                            accd114 =   accc117 + accc118;
                            acce112 =       accd113 + accd114;
                            accc119 = ifmaps3[10][sh + 2][sw + 2] * filter2[fil][10][2][2];
                            acc11 =  acce111 + acce112 + accc119;

                            accc121 = ifmaps3[11][sh + 0][sw + 0] * filter2[fil][11][0][0];
                            accc122 = ifmaps3[11][sh + 0][sw + 1] * filter2[fil][11][0][1];
                            accd121 =   accc121 + accc122;
                            accc123 = ifmaps3[11][sh + 0][sw + 2] * filter2[fil][11][0][2];
                            accc124 = ifmaps3[11][sh + 1][sw + 0] * filter2[fil][11][1][0];
                            accd122 =   accc123 + accc124;
                            acce121 =       accd121 + accd122;
                            accc125 = ifmaps3[11][sh + 1][sw + 1] * filter2[fil][11][1][1];
                            accc126 = ifmaps3[11][sh + 1][sw + 2] * filter2[fil][11][1][2];
                            accd123 =   accc125 + accc126;
                            accc127 = ifmaps3[11][sh + 2][sw + 0] * filter2[fil][11][2][0];
                            accc128 = ifmaps3[11][sh + 2][sw + 1] * filter2[fil][11][2][1];
                            accd124 =   accc127 + accc128;
                            acce122 =       accd123 + accd124;
                            accc129 = ifmaps3[11][sh + 2][sw + 2] * filter2[fil][11][2][2];
                            acc12 =  acce121 + acce122 + accc129;

                            ac6 = acc11 + acc12;
                            a3 = ac5 + ac6;

                            accc131 = ifmaps3[12][sh + 0][sw + 0] * filter2[fil][12][0][0];
                            accc132 = ifmaps3[12][sh + 0][sw + 1] * filter2[fil][12][0][1];
                            accd131 =   accc131 + accc132;
                            accc133 = ifmaps3[12][sh + 0][sw + 2] * filter2[fil][12][0][2];
                            accc134 = ifmaps3[12][sh + 1][sw + 0] * filter2[fil][12][1][0];
                            accd132 =   accc133 + accc134;
                            acce131 =       accd131 + accd132;
                            accc135 = ifmaps3[12][sh + 1][sw + 1] * filter2[fil][12][1][1];
                            accc136 = ifmaps3[12][sh + 1][sw + 2] * filter2[fil][12][1][2];
                            accd133 =   accc135 + accc136;
                            accc137 = ifmaps3[12][sh + 2][sw + 0] * filter2[fil][12][2][0];
                            accc138 = ifmaps3[12][sh + 2][sw + 1] * filter2[fil][12][2][1];
                            accd134 =   accc137 + accc138;
                            acce132 =       accd133 + accd134;
                            accc139 = ifmaps3[12][sh + 2][sw + 2] * filter2[fil][12][2][2];
                            acc13 =  acce131 + acce132 + accc139;

                            accc141 = ifmaps3[13][sh + 0][sw + 0] * filter2[fil][13][0][0];
                            accc142 = ifmaps3[13][sh + 0][sw + 1] * filter2[fil][13][0][1];
                            accd141 =   accc141 + accc142;
                            accc143 = ifmaps3[13][sh + 0][sw + 2] * filter2[fil][13][0][2];
                            accc144 = ifmaps3[13][sh + 1][sw + 0] * filter2[fil][13][1][0];
                            accd142 =   accc143 + accc144;
                            acce141 =       accd141 + accd142;
                            accc145 = ifmaps3[13][sh + 1][sw + 1] * filter2[fil][13][1][1];
                            accc146 = ifmaps3[13][sh + 1][sw + 2] * filter2[fil][13][1][2];
                            accd143 =   accc145 + accc146;
                            accc147 = ifmaps3[13][sh + 2][sw + 0] * filter2[fil][13][2][0];
                            accc148 = ifmaps3[13][sh + 2][sw + 1] * filter2[fil][13][2][1];
                            accd144 =   accc147 + accc148;
                            acce142 =       accd143 + accd144;
                            accc149 = ifmaps3[13][sh + 2][sw + 2] * filter2[fil][13][2][2];
                            acc14 =  acce141 + acce142 + accc149;

                            ac7 = acc13 + acc14;

                            accc151 = ifmaps3[14][sh + 0][sw + 0] * filter2[fil][14][0][0];
                            accc152 = ifmaps3[14][sh + 0][sw + 1] * filter2[fil][14][0][1];
                            accd151 =   accc151 + accc152;
                            accc153 = ifmaps3[14][sh + 0][sw + 2] * filter2[fil][14][0][2];
                            accc154 = ifmaps3[14][sh + 1][sw + 0] * filter2[fil][14][1][0];
                            accd152 =   accc153 + accc154;
                            acce151 =       accd151 + accd152;
                            accc155 = ifmaps3[14][sh + 1][sw + 1] * filter2[fil][14][1][1];
                            accc156 = ifmaps3[14][sh + 1][sw + 2] * filter2[fil][14][1][2];
                            accd153 =   accc155 + accc156;
                            accc157 = ifmaps3[14][sh + 2][sw + 0] * filter2[fil][14][2][0];
                            accc158 = ifmaps3[14][sh + 2][sw + 1] * filter2[fil][14][2][1];
                            accd154 =   accc157 + accc158;
                            acce152 =       accd153 + accd154;
                            accc159 = ifmaps3[14][sh + 2][sw + 2] * filter2[fil][14][2][2];
                            acc15 =  acce151 + acce152 + accc159;

                            accc161 = ifmaps3[15][sh + 0][sw + 0] * filter2[fil][15][0][0];
                            accc162 = ifmaps3[15][sh + 0][sw + 1] * filter2[fil][15][0][1];
                            accd161 =   accc161 + accc162;
                            accc163 = ifmaps3[15][sh + 0][sw + 2] * filter2[fil][15][0][2];
                            accc164 = ifmaps3[15][sh + 1][sw + 0] * filter2[fil][15][1][0];
                            accd162 =   accc163 + accc164;
                            acce161 =       accd161 + accd162;
                            accc165 = ifmaps3[15][sh + 1][sw + 1] * filter2[fil][15][1][1];
                            accc166 = ifmaps3[15][sh + 1][sw + 2] * filter2[fil][15][1][2];
                            accd163 =   accc165 + accc166;
                            accc167 = ifmaps3[15][sh + 2][sw + 0] * filter2[fil][15][2][0];
                            accc168 = ifmaps3[15][sh + 2][sw + 1] * filter2[fil][15][2][1];
                            accd164 =   accc167 + accc168;
                            acce162 =       accd163 + accd164;
                            accc169 = ifmaps3[15][sh + 2][sw + 2] * filter2[fil][15][2][2];
                            acc16 =  acce161 + acce162 + accc169;

                            ac8 = acc15 + acc16;
                            a4 = ac7 + ac8;
                            a_c2 = a3 + a4;

                            // a1 + a2 + a3 + a4;   // a_c1 + a_c2;
                            //acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + + acc10  + acc11  + acc12  + acc13  + acc14  + acc15  + acc16;
                            ofmaps[ofm_idx] += a_c1 + a_c2;
                            ofm_idx += dst_h * dst_w;


                        }   // for fil
                    }   // for sw
                }   // for sh
            }   // for chan

            int icp_ofm = 0;
            for(; icp_ofm < dst_c*dst_h*dst_w; icp_ofm++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                C[icp_ofm] = ofmaps[icp_ofm];
            }
            // actor机制
            if(icp_ofm == dst_c*dst_h*dst_w){
                for(;icp_ofm < ofmelmt;){
                    #ifdef SDSOC
                    #pragma HLS PIPELINE
                    #endif
                    C[icp_ofm] = 0;
                    icp_ofm++;
                }
                //std::cout << "actor on, the last icp_ofm = " << icp_ofm - 1 << std::endl;
            }

            break;
        }

        case CONV6:{
            const int src_c = 192;
            const int src_h = 15;
            const int src_w = 15;
            const int stride = 1;
            const int filter_size = 192*3*3;
            const int filter_h = 3;
            const int filter_w = 3;
            const int dst_c = 192;
            const int dst_h = 13;
            const int dst_w = 13;
            const int end_h = src_h - filter_h + 1;
            const int end_w = src_w - filter_w + 1;

            // 给ofmaps进行初始化
            for(int i = 0; i < dst_c*dst_h*dst_w; i++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                ofmaps[i] = 0;
            }

            for (int icp_ifm = 0, icp_fil = 0,  chan = 0; chan < src_c; chan += fac_para){        // 一个filter的channel维度。其中若chan+=4代表4个filter channel在并行。
                // 用local memory ifmaps3来存储输入。
                for(int cidx = 0; cidx < fac_para; cidx++){
                    for(int hidx = 0; hidx < src_h; hidx++){
                        for(int widx = 0; widx < src_w; widx++){
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            #endif
                            ifmaps3[cidx][hidx][widx] = A[icp_ifm];
                            icp_ifm++;
                        }
                    }
                }
                // actor机制
                if(icp_ifm == src_c*src_h*src_w){
                    for(;icp_ifm < ifmelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif                    
                        actor_1 = A[icp_ifm];
                        icp_ifm++;                              
                    }
                    //std::cout << "actor on, the last icp_ifm = " << icp_ifm - 1 << std::endl;  
                }   
                // 用local memory来存储filter。
                for(int fil = 0; fil < dst_c; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < filter_h; hidx++){
                            for(int widx = 0; widx < filter_w; widx++){
                                #ifdef SDSOC
                                #pragma HLS PIPELINE
                                #endif
                                filter2[fil][cidx][hidx][widx] = B[icp_fil];
                                icp_fil++;
                            }
                        }
                    }
                }
                // actor机制
                if(icp_fil == dst_c*src_c*filter_h*filter_w){
                    for(;icp_fil < filelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif
                        actor_2 = B[icp_fil];
                        icp_fil++;
                    }
                    //std::cout << "actor on, the last icp_fil = " << icp_fil - 1 << std::endl;  
                }  
                // 移窗
                for(int sh = 0; sh < end_h; sh += stride){                  // 纵向移窗。
                    for (int sw = 0; sw < end_w; sw += stride){             // 横向移窗。
                        int ofm_idx = sh/stride * dst_w + sw/stride;
                        for(int fil = 0; fil < dst_c; fil++){      // 切换filter。
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            //#pragma HLS dependence array inter false
                            #endif
                            accc011 = ifmaps3[0][sh + 0][sw + 0] * filter2[fil][0][0][0];
                            accc012 = ifmaps3[0][sh + 0][sw + 1] * filter2[fil][0][0][1];
                            accd011 =   accc011 + accc012;
                            accc013 = ifmaps3[0][sh + 0][sw + 2] * filter2[fil][0][0][2];
                            accc014 = ifmaps3[0][sh + 1][sw + 0] * filter2[fil][0][1][0];
                            accd012 =   accc013 + accc014;
                            acce011 =       accd011 + accd012;
                            accc015 = ifmaps3[0][sh + 1][sw + 1] * filter2[fil][0][1][1];
                            accc016 = ifmaps3[0][sh + 1][sw + 2] * filter2[fil][0][1][2];
                            accd013 =   accc015 + accc016;
                            accc017 = ifmaps3[0][sh + 2][sw + 0] * filter2[fil][0][2][0];
                            accc018 = ifmaps3[0][sh + 2][sw + 1] * filter2[fil][0][2][1];
                            accd014 =   accc017 + accc018;
                            acce012 =       accd013 + accd014;
                            accc019 = ifmaps3[0][sh + 2][sw + 2] * filter2[fil][0][2][2];
                            acc1 =  acce011 + acce012 + accc019;

                            accc021 = ifmaps3[1][sh + 0][sw + 0] * filter2[fil][1][0][0];
                            accc022 = ifmaps3[1][sh + 0][sw + 1] * filter2[fil][1][0][1];
                            accd021 =   accc021 + accc022;
                            accc023 = ifmaps3[1][sh + 0][sw + 2] * filter2[fil][1][0][2];
                            accc024 = ifmaps3[1][sh + 1][sw + 0] * filter2[fil][1][1][0];
                            accd022 =   accc023 + accc024;
                            acce021 =       accd021 + accd022;
                            accc025 = ifmaps3[1][sh + 1][sw + 1] * filter2[fil][1][1][1];
                            accc026 = ifmaps3[1][sh + 1][sw + 2] * filter2[fil][1][1][2];
                            accd023 =   accc025 + accc026;
                            accc027 = ifmaps3[1][sh + 2][sw + 0] * filter2[fil][1][2][0];
                            accc028 = ifmaps3[1][sh + 2][sw + 1] * filter2[fil][1][2][1];
                            accd024 =   accc027 + accc028;
                            acce022 =       accd023 + accd024;
                            accc029 = ifmaps3[1][sh + 2][sw + 2] * filter2[fil][1][2][2];
                            acc2 =  acce021 + acce022 + accc029;

                            ac1 = acc1 + acc2;

                            accc031 = ifmaps3[2][sh + 0][sw + 0] * filter2[fil][2][0][0];
                            accc032 = ifmaps3[2][sh + 0][sw + 1] * filter2[fil][2][0][1];
                            accd031 =   accc031 + accc032;
                            accc033 = ifmaps3[2][sh + 0][sw + 2] * filter2[fil][2][0][2];
                            accc034 = ifmaps3[2][sh + 1][sw + 0] * filter2[fil][2][1][0];
                            accd032 =   accc033 + accc034;
                            acce031 =       accd031 + accd032;
                            accc035 = ifmaps3[2][sh + 1][sw + 1] * filter2[fil][2][1][1];
                            accc036 = ifmaps3[2][sh + 1][sw + 2] * filter2[fil][2][1][2];
                            accd033 =   accc035 + accc036;
                            accc037 = ifmaps3[2][sh + 2][sw + 0] * filter2[fil][2][2][0];
                            accc038 = ifmaps3[2][sh + 2][sw + 1] * filter2[fil][2][2][1];
                            accd034 =   accc037 + accc038;
                            acce032 =       accd033 + accd034;
                            accc039 = ifmaps3[2][sh + 2][sw + 2] * filter2[fil][2][2][2];
                            acc3 =  acce031 + acce032 + accc039;

                            accc041 = ifmaps3[3][sh + 0][sw + 0] * filter2[fil][3][0][0];
                            accc042 = ifmaps3[3][sh + 0][sw + 1] * filter2[fil][3][0][1];
                            accd041 =   accc041 + accc042;
                            accc043 = ifmaps3[3][sh + 0][sw + 2] * filter2[fil][3][0][2];
                            accc044 = ifmaps3[3][sh + 1][sw + 0] * filter2[fil][3][1][0];
                            accd042 =   accc043 + accc044;
                            acce041 =       accd041 + accd042;
                            accc045 = ifmaps3[3][sh + 1][sw + 1] * filter2[fil][3][1][1];
                            accc046 = ifmaps3[3][sh + 1][sw + 2] * filter2[fil][3][1][2];
                            accd043 =   accc045 + accc046;
                            accc047 = ifmaps3[3][sh + 2][sw + 0] * filter2[fil][3][2][0];
                            accc048 = ifmaps3[3][sh + 2][sw + 1] * filter2[fil][3][2][1];
                            accd044 =   accc047 + accc048;
                            acce042 =       accd043 + accd044;
                            accc049 = ifmaps3[3][sh + 2][sw + 2] * filter2[fil][3][2][2];
                            acc4 =  acce041 + acce042 + accc049;

                            ac2 = acc3 +acc4;
                            a1 = ac1 + ac2;

                            accc051 = ifmaps3[4][sh + 0][sw + 0] * filter2[fil][4][0][0];
                            accc052 = ifmaps3[4][sh + 0][sw + 1] * filter2[fil][4][0][1];
                            accd051 =   accc051 + accc052;
                            accc053 = ifmaps3[4][sh + 0][sw + 2] * filter2[fil][4][0][2];
                            accc054 = ifmaps3[4][sh + 1][sw + 0] * filter2[fil][4][1][0];
                            accd052 =   accc053 + accc054;
                            acce051 =       accd051 + accd052;
                            accc055 = ifmaps3[4][sh + 1][sw + 1] * filter2[fil][4][1][1];
                            accc056 = ifmaps3[4][sh + 1][sw + 2] * filter2[fil][4][1][2];
                            accd053 =   accc055 + accc056;
                            accc057 = ifmaps3[4][sh + 2][sw + 0] * filter2[fil][4][2][0];
                            accc058 = ifmaps3[4][sh + 2][sw + 1] * filter2[fil][4][2][1];
                            accd054 =   accc057 + accc058;
                            acce052 =       accd053 + accd054;
                            accc059 = ifmaps3[4][sh + 2][sw + 2] * filter2[fil][4][2][2];
                            acc5 =  acce051 + acce052 + accc059;

                            accc061 = ifmaps3[5][sh + 0][sw + 0] * filter2[fil][5][0][0];
                            accc062 = ifmaps3[5][sh + 0][sw + 1] * filter2[fil][5][0][1];
                            accd061 =   accc061 + accc062;
                            accc063 = ifmaps3[5][sh + 0][sw + 2] * filter2[fil][5][0][2];
                            accc064 = ifmaps3[5][sh + 1][sw + 0] * filter2[fil][5][1][0];
                            accd062 =   accc063 + accc064;
                            acce061 =       accd061 + accd062;
                            accc065 = ifmaps3[5][sh + 1][sw + 1] * filter2[fil][5][1][1];
                            accc066 = ifmaps3[5][sh + 1][sw + 2] * filter2[fil][5][1][2];
                            accd063 =   accc065 + accc066;
                            accc067 = ifmaps3[5][sh + 2][sw + 0] * filter2[fil][5][2][0];
                            accc068 = ifmaps3[5][sh + 2][sw + 1] * filter2[fil][5][2][1];
                            accd064 =   accc067 + accc068;
                            acce062 =       accd063 + accd064;
                            accc069 = ifmaps3[5][sh + 2][sw + 2] * filter2[fil][5][2][2];
                            acc6 =  acce061 + acce062 + accc069;

                            ac3 = acc5 + acc6;

                            accc071 = ifmaps3[6][sh + 0][sw + 0] * filter2[fil][6][0][0];
                            accc072 = ifmaps3[6][sh + 0][sw + 1] * filter2[fil][6][0][1];
                            accd071 =   accc071 + accc072;
                            accc073 = ifmaps3[6][sh + 0][sw + 2] * filter2[fil][6][0][2];
                            accc074 = ifmaps3[6][sh + 1][sw + 0] * filter2[fil][6][1][0];
                            accd072 =   accc073 + accc074;
                            acce071 =       accd071 + accd072;
                            accc075 = ifmaps3[6][sh + 1][sw + 1] * filter2[fil][6][1][1];
                            accc076 = ifmaps3[6][sh + 1][sw + 2] * filter2[fil][6][1][2];
                            accd073 =   accc075 + accc076;
                            accc077 = ifmaps3[6][sh + 2][sw + 0] * filter2[fil][6][2][0];
                            accc078 = ifmaps3[6][sh + 2][sw + 1] * filter2[fil][6][2][1];
                            accd074 =   accc077 + accc078;
                            acce072 =       accd073 + accd074;
                            accc079 = ifmaps3[6][sh + 2][sw + 2] * filter2[fil][6][2][2];
                            acc7 =  acce071 + acce072 + accc079;

                            accc081 = ifmaps3[7][sh + 0][sw + 0] * filter2[fil][7][0][0];
                            accc082 = ifmaps3[7][sh + 0][sw + 1] * filter2[fil][7][0][1];
                            accd081 =   accc081 + accc082;
                            accc083 = ifmaps3[7][sh + 0][sw + 2] * filter2[fil][7][0][2];
                            accc084 = ifmaps3[7][sh + 1][sw + 0] * filter2[fil][7][1][0];
                            accd082 =   accc083 + accc084;
                            acce081 =       accd081 + accd082;
                            accc085 = ifmaps3[7][sh + 1][sw + 1] * filter2[fil][7][1][1];
                            accc086 = ifmaps3[7][sh + 1][sw + 2] * filter2[fil][7][1][2];
                            accd083 =   accc085 + accc086;
                            accc087 = ifmaps3[7][sh + 2][sw + 0] * filter2[fil][7][2][0];
                            accc088 = ifmaps3[7][sh + 2][sw + 1] * filter2[fil][7][2][1];
                            accd084 =   accc087 + accc088;
                            acce082 =       accd083 + accd084;
                            accc089 = ifmaps3[7][sh + 2][sw + 2] * filter2[fil][7][2][2];
                            acc8 =  acce081 + acce082 + accc089;

                            ac4 = acc7 + acc8;
                            a2 = ac3 + ac4;
                            a_c1 = a1 + a2;

                            accc091 = ifmaps3[8][sh + 0][sw + 0] * filter2[fil][8][0][0];
                            accc092 = ifmaps3[8][sh + 0][sw + 1] * filter2[fil][8][0][1];
                            accd091 =   accc091 + accc092;
                            accc093 = ifmaps3[8][sh + 0][sw + 2] * filter2[fil][8][0][2];
                            accc094 = ifmaps3[8][sh + 1][sw + 0] * filter2[fil][8][1][0];
                            accd092 =   accc093 + accc094;
                            acce091 =       accd091 + accd092;
                            accc095 = ifmaps3[8][sh + 1][sw + 1] * filter2[fil][8][1][1];
                            accc096 = ifmaps3[8][sh + 1][sw + 2] * filter2[fil][8][1][2];
                            accd093 =   accc095 + accc096;
                            accc097 = ifmaps3[8][sh + 2][sw + 0] * filter2[fil][8][2][0];
                            accc098 = ifmaps3[8][sh + 2][sw + 1] * filter2[fil][8][2][1];
                            accd094 =   accc097 + accc098;
                            acce092 =       accd093 + accd094;
                            accc099 = ifmaps3[8][sh + 2][sw + 2] * filter2[fil][8][2][2];
                            acc9 =  acce091 + acce092 + accc099;

                            accc101 = ifmaps3[9][sh + 0][sw + 0] * filter2[fil][9][0][0];
                            accc102 = ifmaps3[9][sh + 0][sw + 1] * filter2[fil][9][0][1];
                            accd101 =   accc101 + accc102;
                            accc103 = ifmaps3[9][sh + 0][sw + 2] * filter2[fil][9][0][2];
                            accc104 = ifmaps3[9][sh + 1][sw + 0] * filter2[fil][9][1][0];
                            accd102 =   accc103 + accc104;
                            acce101 =       accd101 + accd102;
                            accc105 = ifmaps3[9][sh + 1][sw + 1] * filter2[fil][9][1][1];
                            accc106 = ifmaps3[9][sh + 1][sw + 2] * filter2[fil][9][1][2];
                            accd103 =   accc105 + accc106;
                            accc107 = ifmaps3[9][sh + 2][sw + 0] * filter2[fil][9][2][0];
                            accc108 = ifmaps3[9][sh + 2][sw + 1] * filter2[fil][9][2][1];
                            accd104 =   accc107 + accc108;
                            acce102 =       accd103 + accd104;
                            accc109 = ifmaps3[9][sh + 2][sw + 2] * filter2[fil][9][2][2];
                            acc10 =  acce101 + acce102 + accc109;

                            ac5 = acc9 + acc10;

                            accc111 = ifmaps3[10][sh + 0][sw + 0] * filter2[fil][10][0][0];
                            accc112 = ifmaps3[10][sh + 0][sw + 1] * filter2[fil][10][0][1];
                            accd111 =   accc111 + accc112;
                            accc113 = ifmaps3[10][sh + 0][sw + 2] * filter2[fil][10][0][2];
                            accc114 = ifmaps3[10][sh + 1][sw + 0] * filter2[fil][10][1][0];
                            accd112 =   accc113 + accc114;
                            acce111 =       accd111 + accd112;
                            accc115 = ifmaps3[10][sh + 1][sw + 1] * filter2[fil][10][1][1];
                            accc116 = ifmaps3[10][sh + 1][sw + 2] * filter2[fil][10][1][2];
                            accd113 =   accc115 + accc116;
                            accc117 = ifmaps3[10][sh + 2][sw + 0] * filter2[fil][10][2][0];
                            accc118 = ifmaps3[10][sh + 2][sw + 1] * filter2[fil][10][2][1];
                            accd114 =   accc117 + accc118;
                            acce112 =       accd113 + accd114;
                            accc119 = ifmaps3[10][sh + 2][sw + 2] * filter2[fil][10][2][2];
                            acc11 =  acce111 + acce112 + accc119;

                            accc121 = ifmaps3[11][sh + 0][sw + 0] * filter2[fil][11][0][0];
                            accc122 = ifmaps3[11][sh + 0][sw + 1] * filter2[fil][11][0][1];
                            accd121 =   accc121 + accc122;
                            accc123 = ifmaps3[11][sh + 0][sw + 2] * filter2[fil][11][0][2];
                            accc124 = ifmaps3[11][sh + 1][sw + 0] * filter2[fil][11][1][0];
                            accd122 =   accc123 + accc124;
                            acce121 =       accd121 + accd122;
                            accc125 = ifmaps3[11][sh + 1][sw + 1] * filter2[fil][11][1][1];
                            accc126 = ifmaps3[11][sh + 1][sw + 2] * filter2[fil][11][1][2];
                            accd123 =   accc125 + accc126;
                            accc127 = ifmaps3[11][sh + 2][sw + 0] * filter2[fil][11][2][0];
                            accc128 = ifmaps3[11][sh + 2][sw + 1] * filter2[fil][11][2][1];
                            accd124 =   accc127 + accc128;
                            acce122 =       accd123 + accd124;
                            accc129 = ifmaps3[11][sh + 2][sw + 2] * filter2[fil][11][2][2];
                            acc12 =  acce121 + acce122 + accc129;

                            ac6 = acc11 + acc12;
                            a3 = ac5 + ac6;

                            accc131 = ifmaps3[12][sh + 0][sw + 0] * filter2[fil][12][0][0];
                            accc132 = ifmaps3[12][sh + 0][sw + 1] * filter2[fil][12][0][1];
                            accd131 =   accc131 + accc132;
                            accc133 = ifmaps3[12][sh + 0][sw + 2] * filter2[fil][12][0][2];
                            accc134 = ifmaps3[12][sh + 1][sw + 0] * filter2[fil][12][1][0];
                            accd132 =   accc133 + accc134;
                            acce131 =       accd131 + accd132;
                            accc135 = ifmaps3[12][sh + 1][sw + 1] * filter2[fil][12][1][1];
                            accc136 = ifmaps3[12][sh + 1][sw + 2] * filter2[fil][12][1][2];
                            accd133 =   accc135 + accc136;
                            accc137 = ifmaps3[12][sh + 2][sw + 0] * filter2[fil][12][2][0];
                            accc138 = ifmaps3[12][sh + 2][sw + 1] * filter2[fil][12][2][1];
                            accd134 =   accc137 + accc138;
                            acce132 =       accd133 + accd134;
                            accc139 = ifmaps3[12][sh + 2][sw + 2] * filter2[fil][12][2][2];
                            acc13 =  acce131 + acce132 + accc139;

                            accc141 = ifmaps3[13][sh + 0][sw + 0] * filter2[fil][13][0][0];
                            accc142 = ifmaps3[13][sh + 0][sw + 1] * filter2[fil][13][0][1];
                            accd141 =   accc141 + accc142;
                            accc143 = ifmaps3[13][sh + 0][sw + 2] * filter2[fil][13][0][2];
                            accc144 = ifmaps3[13][sh + 1][sw + 0] * filter2[fil][13][1][0];
                            accd142 =   accc143 + accc144;
                            acce141 =       accd141 + accd142;
                            accc145 = ifmaps3[13][sh + 1][sw + 1] * filter2[fil][13][1][1];
                            accc146 = ifmaps3[13][sh + 1][sw + 2] * filter2[fil][13][1][2];
                            accd143 =   accc145 + accc146;
                            accc147 = ifmaps3[13][sh + 2][sw + 0] * filter2[fil][13][2][0];
                            accc148 = ifmaps3[13][sh + 2][sw + 1] * filter2[fil][13][2][1];
                            accd144 =   accc147 + accc148;
                            acce142 =       accd143 + accd144;
                            accc149 = ifmaps3[13][sh + 2][sw + 2] * filter2[fil][13][2][2];
                            acc14 =  acce141 + acce142 + accc149;

                            ac7 = acc13 + acc14;

                            accc151 = ifmaps3[14][sh + 0][sw + 0] * filter2[fil][14][0][0];
                            accc152 = ifmaps3[14][sh + 0][sw + 1] * filter2[fil][14][0][1];
                            accd151 =   accc151 + accc152;
                            accc153 = ifmaps3[14][sh + 0][sw + 2] * filter2[fil][14][0][2];
                            accc154 = ifmaps3[14][sh + 1][sw + 0] * filter2[fil][14][1][0];
                            accd152 =   accc153 + accc154;
                            acce151 =       accd151 + accd152;
                            accc155 = ifmaps3[14][sh + 1][sw + 1] * filter2[fil][14][1][1];
                            accc156 = ifmaps3[14][sh + 1][sw + 2] * filter2[fil][14][1][2];
                            accd153 =   accc155 + accc156;
                            accc157 = ifmaps3[14][sh + 2][sw + 0] * filter2[fil][14][2][0];
                            accc158 = ifmaps3[14][sh + 2][sw + 1] * filter2[fil][14][2][1];
                            accd154 =   accc157 + accc158;
                            acce152 =       accd153 + accd154;
                            accc159 = ifmaps3[14][sh + 2][sw + 2] * filter2[fil][14][2][2];
                            acc15 =  acce151 + acce152 + accc159;

                            accc161 = ifmaps3[15][sh + 0][sw + 0] * filter2[fil][15][0][0];
                            accc162 = ifmaps3[15][sh + 0][sw + 1] * filter2[fil][15][0][1];
                            accd161 =   accc161 + accc162;
                            accc163 = ifmaps3[15][sh + 0][sw + 2] * filter2[fil][15][0][2];
                            accc164 = ifmaps3[15][sh + 1][sw + 0] * filter2[fil][15][1][0];
                            accd162 =   accc163 + accc164;
                            acce161 =       accd161 + accd162;
                            accc165 = ifmaps3[15][sh + 1][sw + 1] * filter2[fil][15][1][1];
                            accc166 = ifmaps3[15][sh + 1][sw + 2] * filter2[fil][15][1][2];
                            accd163 =   accc165 + accc166;
                            accc167 = ifmaps3[15][sh + 2][sw + 0] * filter2[fil][15][2][0];
                            accc168 = ifmaps3[15][sh + 2][sw + 1] * filter2[fil][15][2][1];
                            accd164 =   accc167 + accc168;
                            acce162 =       accd163 + accd164;
                            accc169 = ifmaps3[15][sh + 2][sw + 2] * filter2[fil][15][2][2];
                            acc16 =  acce161 + acce162 + accc169;

                            ac8 = acc15 + acc16;
                            a4 = ac7 + ac8;
                            a_c2 = a3 + a4;

                            // a1 + a2 + a3 + a4;   // a_c1 + a_c2;
                            //acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + + acc10  + acc11  + acc12  + acc13  + acc14  + acc15  + acc16;
                            ofmaps[ofm_idx] += a_c1 + a_c2;
                            ofm_idx += dst_h * dst_w;


                        }   // for fil
                    }   // for sw
                }   // for sh
            }   // for chan

            int icp_ofm = 0;
            for(; icp_ofm < dst_c*dst_h*dst_w; icp_ofm++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                C[icp_ofm] = ofmaps[icp_ofm];
            }
            // actor机制
            if(icp_ofm == dst_c*dst_h*dst_w){
                for(;icp_ofm < ofmelmt;){
                    #ifdef SDSOC
                    #pragma HLS PIPELINE
                    #endif
                    C[icp_ofm] = 0;
                    icp_ofm++;
                }
                //std::cout << "actor on, the last icp_ofm = " << icp_ofm - 1 << std::endl;
            }

            break;
        }

        case CONV7:{
            const int src_c = 192;
            const int src_h = 15;
            const int src_w = 15;
            const int stride = 1;
            const int filter_size = 192*3*3;
            const int filter_h = 3;
            const int filter_w = 3;
            const int dst_c = 128;
            const int dst_h = 13;
            const int dst_w = 13;
            const int end_h = src_h - filter_h + 1;
            const int end_w = src_w - filter_w + 1;

            // 给ofmaps进行初始化
            for(int i = 0; i < dst_c*dst_h*dst_w; i++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                ofmaps[i] = 0;
            }

            for (int icp_ifm = 0, icp_fil = 0,  chan = 0; chan < src_c; chan += fac_para){        // 一个filter的channel维度。其中若chan+=4代表4个filter channel在并行。
                // 用local memory ifmaps3来存储输入。
                for(int cidx = 0; cidx < fac_para; cidx++){
                    for(int hidx = 0; hidx < src_h; hidx++){
                        for(int widx = 0; widx < src_w; widx++){
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            #endif
                            ifmaps3[cidx][hidx][widx] = A[icp_ifm];
                            icp_ifm++;
                        }
                    }
                }
                // actor机制
                if(icp_ifm == src_c*src_h*src_w){
                    for(;icp_ifm < ifmelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif                    
                        actor_1 = A[icp_ifm];
                        icp_ifm++;                              
                    }
                    //std::cout << "actor on, the last icp_ifm = " << icp_ifm - 1 << std::endl;  
                }   
                // 用local memory来存储filter。
                for(int fil = 0; fil < dst_c; fil++){
                    for(int cidx = 0; cidx < fac_para; cidx++){
                        for(int hidx = 0; hidx < filter_h; hidx++){
                            for(int widx = 0; widx < filter_w; widx++){
                                #ifdef SDSOC
                                #pragma HLS PIPELINE
                                #endif
                                filter2[fil][cidx][hidx][widx] = B[icp_fil];
                                icp_fil++;
                            }
                        }
                    }
                }
                // actor机制
                if(icp_fil == dst_c*src_c*filter_h*filter_w){
                    for(;icp_fil < filelmt;){
                        #ifdef SDSOC
                        #pragma HLS PIPELINE
                        #endif
                        actor_2 = B[icp_fil];
                        icp_fil++;
                    }
                    //std::cout << "actor on, the last icp_fil = " << icp_fil - 1 << std::endl;  
                }  
                // 移窗
                for(int sh = 0; sh < end_h; sh += stride){                  // 纵向移窗。
                    for (int sw = 0; sw < end_w; sw += stride){             // 横向移窗。
                        int ofm_idx = sh/stride * dst_w + sw/stride;
                        for(int fil = 0; fil < dst_c; fil++){      // 切换filter。
                            #ifdef SDSOC
                            #pragma HLS PIPELINE
                            //#pragma HLS dependence array inter false
                            #endif
                            accc011 = ifmaps3[0][sh + 0][sw + 0] * filter2[fil][0][0][0];
                            accc012 = ifmaps3[0][sh + 0][sw + 1] * filter2[fil][0][0][1];
                            accd011 =   accc011 + accc012;
                            accc013 = ifmaps3[0][sh + 0][sw + 2] * filter2[fil][0][0][2];
                            accc014 = ifmaps3[0][sh + 1][sw + 0] * filter2[fil][0][1][0];
                            accd012 =   accc013 + accc014;
                            acce011 =       accd011 + accd012;
                            accc015 = ifmaps3[0][sh + 1][sw + 1] * filter2[fil][0][1][1];
                            accc016 = ifmaps3[0][sh + 1][sw + 2] * filter2[fil][0][1][2];
                            accd013 =   accc015 + accc016;
                            accc017 = ifmaps3[0][sh + 2][sw + 0] * filter2[fil][0][2][0];
                            accc018 = ifmaps3[0][sh + 2][sw + 1] * filter2[fil][0][2][1];
                            accd014 =   accc017 + accc018;
                            acce012 =       accd013 + accd014;
                            accc019 = ifmaps3[0][sh + 2][sw + 2] * filter2[fil][0][2][2];
                            acc1 =  acce011 + acce012 + accc019;

                            accc021 = ifmaps3[1][sh + 0][sw + 0] * filter2[fil][1][0][0];
                            accc022 = ifmaps3[1][sh + 0][sw + 1] * filter2[fil][1][0][1];
                            accd021 =   accc021 + accc022;
                            accc023 = ifmaps3[1][sh + 0][sw + 2] * filter2[fil][1][0][2];
                            accc024 = ifmaps3[1][sh + 1][sw + 0] * filter2[fil][1][1][0];
                            accd022 =   accc023 + accc024;
                            acce021 =       accd021 + accd022;
                            accc025 = ifmaps3[1][sh + 1][sw + 1] * filter2[fil][1][1][1];
                            accc026 = ifmaps3[1][sh + 1][sw + 2] * filter2[fil][1][1][2];
                            accd023 =   accc025 + accc026;
                            accc027 = ifmaps3[1][sh + 2][sw + 0] * filter2[fil][1][2][0];
                            accc028 = ifmaps3[1][sh + 2][sw + 1] * filter2[fil][1][2][1];
                            accd024 =   accc027 + accc028;
                            acce022 =       accd023 + accd024;
                            accc029 = ifmaps3[1][sh + 2][sw + 2] * filter2[fil][1][2][2];
                            acc2 =  acce021 + acce022 + accc029;

                            ac1 = acc1 + acc2;

                            accc031 = ifmaps3[2][sh + 0][sw + 0] * filter2[fil][2][0][0];
                            accc032 = ifmaps3[2][sh + 0][sw + 1] * filter2[fil][2][0][1];
                            accd031 =   accc031 + accc032;
                            accc033 = ifmaps3[2][sh + 0][sw + 2] * filter2[fil][2][0][2];
                            accc034 = ifmaps3[2][sh + 1][sw + 0] * filter2[fil][2][1][0];
                            accd032 =   accc033 + accc034;
                            acce031 =       accd031 + accd032;
                            accc035 = ifmaps3[2][sh + 1][sw + 1] * filter2[fil][2][1][1];
                            accc036 = ifmaps3[2][sh + 1][sw + 2] * filter2[fil][2][1][2];
                            accd033 =   accc035 + accc036;
                            accc037 = ifmaps3[2][sh + 2][sw + 0] * filter2[fil][2][2][0];
                            accc038 = ifmaps3[2][sh + 2][sw + 1] * filter2[fil][2][2][1];
                            accd034 =   accc037 + accc038;
                            acce032 =       accd033 + accd034;
                            accc039 = ifmaps3[2][sh + 2][sw + 2] * filter2[fil][2][2][2];
                            acc3 =  acce031 + acce032 + accc039;

                            accc041 = ifmaps3[3][sh + 0][sw + 0] * filter2[fil][3][0][0];
                            accc042 = ifmaps3[3][sh + 0][sw + 1] * filter2[fil][3][0][1];
                            accd041 =   accc041 + accc042;
                            accc043 = ifmaps3[3][sh + 0][sw + 2] * filter2[fil][3][0][2];
                            accc044 = ifmaps3[3][sh + 1][sw + 0] * filter2[fil][3][1][0];
                            accd042 =   accc043 + accc044;
                            acce041 =       accd041 + accd042;
                            accc045 = ifmaps3[3][sh + 1][sw + 1] * filter2[fil][3][1][1];
                            accc046 = ifmaps3[3][sh + 1][sw + 2] * filter2[fil][3][1][2];
                            accd043 =   accc045 + accc046;
                            accc047 = ifmaps3[3][sh + 2][sw + 0] * filter2[fil][3][2][0];
                            accc048 = ifmaps3[3][sh + 2][sw + 1] * filter2[fil][3][2][1];
                            accd044 =   accc047 + accc048;
                            acce042 =       accd043 + accd044;
                            accc049 = ifmaps3[3][sh + 2][sw + 2] * filter2[fil][3][2][2];
                            acc4 =  acce041 + acce042 + accc049;

                            ac2 = acc3 +acc4;
                            a1 = ac1 + ac2;

                            accc051 = ifmaps3[4][sh + 0][sw + 0] * filter2[fil][4][0][0];
                            accc052 = ifmaps3[4][sh + 0][sw + 1] * filter2[fil][4][0][1];
                            accd051 =   accc051 + accc052;
                            accc053 = ifmaps3[4][sh + 0][sw + 2] * filter2[fil][4][0][2];
                            accc054 = ifmaps3[4][sh + 1][sw + 0] * filter2[fil][4][1][0];
                            accd052 =   accc053 + accc054;
                            acce051 =       accd051 + accd052;
                            accc055 = ifmaps3[4][sh + 1][sw + 1] * filter2[fil][4][1][1];
                            accc056 = ifmaps3[4][sh + 1][sw + 2] * filter2[fil][4][1][2];
                            accd053 =   accc055 + accc056;
                            accc057 = ifmaps3[4][sh + 2][sw + 0] * filter2[fil][4][2][0];
                            accc058 = ifmaps3[4][sh + 2][sw + 1] * filter2[fil][4][2][1];
                            accd054 =   accc057 + accc058;
                            acce052 =       accd053 + accd054;
                            accc059 = ifmaps3[4][sh + 2][sw + 2] * filter2[fil][4][2][2];
                            acc5 =  acce051 + acce052 + accc059;

                            accc061 = ifmaps3[5][sh + 0][sw + 0] * filter2[fil][5][0][0];
                            accc062 = ifmaps3[5][sh + 0][sw + 1] * filter2[fil][5][0][1];
                            accd061 =   accc061 + accc062;
                            accc063 = ifmaps3[5][sh + 0][sw + 2] * filter2[fil][5][0][2];
                            accc064 = ifmaps3[5][sh + 1][sw + 0] * filter2[fil][5][1][0];
                            accd062 =   accc063 + accc064;
                            acce061 =       accd061 + accd062;
                            accc065 = ifmaps3[5][sh + 1][sw + 1] * filter2[fil][5][1][1];
                            accc066 = ifmaps3[5][sh + 1][sw + 2] * filter2[fil][5][1][2];
                            accd063 =   accc065 + accc066;
                            accc067 = ifmaps3[5][sh + 2][sw + 0] * filter2[fil][5][2][0];
                            accc068 = ifmaps3[5][sh + 2][sw + 1] * filter2[fil][5][2][1];
                            accd064 =   accc067 + accc068;
                            acce062 =       accd063 + accd064;
                            accc069 = ifmaps3[5][sh + 2][sw + 2] * filter2[fil][5][2][2];
                            acc6 =  acce061 + acce062 + accc069;

                            ac3 = acc5 + acc6;

                            accc071 = ifmaps3[6][sh + 0][sw + 0] * filter2[fil][6][0][0];
                            accc072 = ifmaps3[6][sh + 0][sw + 1] * filter2[fil][6][0][1];
                            accd071 =   accc071 + accc072;
                            accc073 = ifmaps3[6][sh + 0][sw + 2] * filter2[fil][6][0][2];
                            accc074 = ifmaps3[6][sh + 1][sw + 0] * filter2[fil][6][1][0];
                            accd072 =   accc073 + accc074;
                            acce071 =       accd071 + accd072;
                            accc075 = ifmaps3[6][sh + 1][sw + 1] * filter2[fil][6][1][1];
                            accc076 = ifmaps3[6][sh + 1][sw + 2] * filter2[fil][6][1][2];
                            accd073 =   accc075 + accc076;
                            accc077 = ifmaps3[6][sh + 2][sw + 0] * filter2[fil][6][2][0];
                            accc078 = ifmaps3[6][sh + 2][sw + 1] * filter2[fil][6][2][1];
                            accd074 =   accc077 + accc078;
                            acce072 =       accd073 + accd074;
                            accc079 = ifmaps3[6][sh + 2][sw + 2] * filter2[fil][6][2][2];
                            acc7 =  acce071 + acce072 + accc079;

                            accc081 = ifmaps3[7][sh + 0][sw + 0] * filter2[fil][7][0][0];
                            accc082 = ifmaps3[7][sh + 0][sw + 1] * filter2[fil][7][0][1];
                            accd081 =   accc081 + accc082;
                            accc083 = ifmaps3[7][sh + 0][sw + 2] * filter2[fil][7][0][2];
                            accc084 = ifmaps3[7][sh + 1][sw + 0] * filter2[fil][7][1][0];
                            accd082 =   accc083 + accc084;
                            acce081 =       accd081 + accd082;
                            accc085 = ifmaps3[7][sh + 1][sw + 1] * filter2[fil][7][1][1];
                            accc086 = ifmaps3[7][sh + 1][sw + 2] * filter2[fil][7][1][2];
                            accd083 =   accc085 + accc086;
                            accc087 = ifmaps3[7][sh + 2][sw + 0] * filter2[fil][7][2][0];
                            accc088 = ifmaps3[7][sh + 2][sw + 1] * filter2[fil][7][2][1];
                            accd084 =   accc087 + accc088;
                            acce082 =       accd083 + accd084;
                            accc089 = ifmaps3[7][sh + 2][sw + 2] * filter2[fil][7][2][2];
                            acc8 =  acce081 + acce082 + accc089;

                            ac4 = acc7 + acc8;
                            a2 = ac3 + ac4;
                            a_c1 = a1 + a2;

                            accc091 = ifmaps3[8][sh + 0][sw + 0] * filter2[fil][8][0][0];
                            accc092 = ifmaps3[8][sh + 0][sw + 1] * filter2[fil][8][0][1];
                            accd091 =   accc091 + accc092;
                            accc093 = ifmaps3[8][sh + 0][sw + 2] * filter2[fil][8][0][2];
                            accc094 = ifmaps3[8][sh + 1][sw + 0] * filter2[fil][8][1][0];
                            accd092 =   accc093 + accc094;
                            acce091 =       accd091 + accd092;
                            accc095 = ifmaps3[8][sh + 1][sw + 1] * filter2[fil][8][1][1];
                            accc096 = ifmaps3[8][sh + 1][sw + 2] * filter2[fil][8][1][2];
                            accd093 =   accc095 + accc096;
                            accc097 = ifmaps3[8][sh + 2][sw + 0] * filter2[fil][8][2][0];
                            accc098 = ifmaps3[8][sh + 2][sw + 1] * filter2[fil][8][2][1];
                            accd094 =   accc097 + accc098;
                            acce092 =       accd093 + accd094;
                            accc099 = ifmaps3[8][sh + 2][sw + 2] * filter2[fil][8][2][2];
                            acc9 =  acce091 + acce092 + accc099;

                            accc101 = ifmaps3[9][sh + 0][sw + 0] * filter2[fil][9][0][0];
                            accc102 = ifmaps3[9][sh + 0][sw + 1] * filter2[fil][9][0][1];
                            accd101 =   accc101 + accc102;
                            accc103 = ifmaps3[9][sh + 0][sw + 2] * filter2[fil][9][0][2];
                            accc104 = ifmaps3[9][sh + 1][sw + 0] * filter2[fil][9][1][0];
                            accd102 =   accc103 + accc104;
                            acce101 =       accd101 + accd102;
                            accc105 = ifmaps3[9][sh + 1][sw + 1] * filter2[fil][9][1][1];
                            accc106 = ifmaps3[9][sh + 1][sw + 2] * filter2[fil][9][1][2];
                            accd103 =   accc105 + accc106;
                            accc107 = ifmaps3[9][sh + 2][sw + 0] * filter2[fil][9][2][0];
                            accc108 = ifmaps3[9][sh + 2][sw + 1] * filter2[fil][9][2][1];
                            accd104 =   accc107 + accc108;
                            acce102 =       accd103 + accd104;
                            accc109 = ifmaps3[9][sh + 2][sw + 2] * filter2[fil][9][2][2];
                            acc10 =  acce101 + acce102 + accc109;

                            ac5 = acc9 + acc10;

                            accc111 = ifmaps3[10][sh + 0][sw + 0] * filter2[fil][10][0][0];
                            accc112 = ifmaps3[10][sh + 0][sw + 1] * filter2[fil][10][0][1];
                            accd111 =   accc111 + accc112;
                            accc113 = ifmaps3[10][sh + 0][sw + 2] * filter2[fil][10][0][2];
                            accc114 = ifmaps3[10][sh + 1][sw + 0] * filter2[fil][10][1][0];
                            accd112 =   accc113 + accc114;
                            acce111 =       accd111 + accd112;
                            accc115 = ifmaps3[10][sh + 1][sw + 1] * filter2[fil][10][1][1];
                            accc116 = ifmaps3[10][sh + 1][sw + 2] * filter2[fil][10][1][2];
                            accd113 =   accc115 + accc116;
                            accc117 = ifmaps3[10][sh + 2][sw + 0] * filter2[fil][10][2][0];
                            accc118 = ifmaps3[10][sh + 2][sw + 1] * filter2[fil][10][2][1];
                            accd114 =   accc117 + accc118;
                            acce112 =       accd113 + accd114;
                            accc119 = ifmaps3[10][sh + 2][sw + 2] * filter2[fil][10][2][2];
                            acc11 =  acce111 + acce112 + accc119;

                            accc121 = ifmaps3[11][sh + 0][sw + 0] * filter2[fil][11][0][0];
                            accc122 = ifmaps3[11][sh + 0][sw + 1] * filter2[fil][11][0][1];
                            accd121 =   accc121 + accc122;
                            accc123 = ifmaps3[11][sh + 0][sw + 2] * filter2[fil][11][0][2];
                            accc124 = ifmaps3[11][sh + 1][sw + 0] * filter2[fil][11][1][0];
                            accd122 =   accc123 + accc124;
                            acce121 =       accd121 + accd122;
                            accc125 = ifmaps3[11][sh + 1][sw + 1] * filter2[fil][11][1][1];
                            accc126 = ifmaps3[11][sh + 1][sw + 2] * filter2[fil][11][1][2];
                            accd123 =   accc125 + accc126;
                            accc127 = ifmaps3[11][sh + 2][sw + 0] * filter2[fil][11][2][0];
                            accc128 = ifmaps3[11][sh + 2][sw + 1] * filter2[fil][11][2][1];
                            accd124 =   accc127 + accc128;
                            acce122 =       accd123 + accd124;
                            accc129 = ifmaps3[11][sh + 2][sw + 2] * filter2[fil][11][2][2];
                            acc12 =  acce121 + acce122 + accc129;

                            ac6 = acc11 + acc12;
                            a3 = ac5 + ac6;

                            accc131 = ifmaps3[12][sh + 0][sw + 0] * filter2[fil][12][0][0];
                            accc132 = ifmaps3[12][sh + 0][sw + 1] * filter2[fil][12][0][1];
                            accd131 =   accc131 + accc132;
                            accc133 = ifmaps3[12][sh + 0][sw + 2] * filter2[fil][12][0][2];
                            accc134 = ifmaps3[12][sh + 1][sw + 0] * filter2[fil][12][1][0];
                            accd132 =   accc133 + accc134;
                            acce131 =       accd131 + accd132;
                            accc135 = ifmaps3[12][sh + 1][sw + 1] * filter2[fil][12][1][1];
                            accc136 = ifmaps3[12][sh + 1][sw + 2] * filter2[fil][12][1][2];
                            accd133 =   accc135 + accc136;
                            accc137 = ifmaps3[12][sh + 2][sw + 0] * filter2[fil][12][2][0];
                            accc138 = ifmaps3[12][sh + 2][sw + 1] * filter2[fil][12][2][1];
                            accd134 =   accc137 + accc138;
                            acce132 =       accd133 + accd134;
                            accc139 = ifmaps3[12][sh + 2][sw + 2] * filter2[fil][12][2][2];
                            acc13 =  acce131 + acce132 + accc139;

                            accc141 = ifmaps3[13][sh + 0][sw + 0] * filter2[fil][13][0][0];
                            accc142 = ifmaps3[13][sh + 0][sw + 1] * filter2[fil][13][0][1];
                            accd141 =   accc141 + accc142;
                            accc143 = ifmaps3[13][sh + 0][sw + 2] * filter2[fil][13][0][2];
                            accc144 = ifmaps3[13][sh + 1][sw + 0] * filter2[fil][13][1][0];
                            accd142 =   accc143 + accc144;
                            acce141 =       accd141 + accd142;
                            accc145 = ifmaps3[13][sh + 1][sw + 1] * filter2[fil][13][1][1];
                            accc146 = ifmaps3[13][sh + 1][sw + 2] * filter2[fil][13][1][2];
                            accd143 =   accc145 + accc146;
                            accc147 = ifmaps3[13][sh + 2][sw + 0] * filter2[fil][13][2][0];
                            accc148 = ifmaps3[13][sh + 2][sw + 1] * filter2[fil][13][2][1];
                            accd144 =   accc147 + accc148;
                            acce142 =       accd143 + accd144;
                            accc149 = ifmaps3[13][sh + 2][sw + 2] * filter2[fil][13][2][2];
                            acc14 =  acce141 + acce142 + accc149;

                            ac7 = acc13 + acc14;

                            accc151 = ifmaps3[14][sh + 0][sw + 0] * filter2[fil][14][0][0];
                            accc152 = ifmaps3[14][sh + 0][sw + 1] * filter2[fil][14][0][1];
                            accd151 =   accc151 + accc152;
                            accc153 = ifmaps3[14][sh + 0][sw + 2] * filter2[fil][14][0][2];
                            accc154 = ifmaps3[14][sh + 1][sw + 0] * filter2[fil][14][1][0];
                            accd152 =   accc153 + accc154;
                            acce151 =       accd151 + accd152;
                            accc155 = ifmaps3[14][sh + 1][sw + 1] * filter2[fil][14][1][1];
                            accc156 = ifmaps3[14][sh + 1][sw + 2] * filter2[fil][14][1][2];
                            accd153 =   accc155 + accc156;
                            accc157 = ifmaps3[14][sh + 2][sw + 0] * filter2[fil][14][2][0];
                            accc158 = ifmaps3[14][sh + 2][sw + 1] * filter2[fil][14][2][1];
                            accd154 =   accc157 + accc158;
                            acce152 =       accd153 + accd154;
                            accc159 = ifmaps3[14][sh + 2][sw + 2] * filter2[fil][14][2][2];
                            acc15 =  acce151 + acce152 + accc159;

                            accc161 = ifmaps3[15][sh + 0][sw + 0] * filter2[fil][15][0][0];
                            accc162 = ifmaps3[15][sh + 0][sw + 1] * filter2[fil][15][0][1];
                            accd161 =   accc161 + accc162;
                            accc163 = ifmaps3[15][sh + 0][sw + 2] * filter2[fil][15][0][2];
                            accc164 = ifmaps3[15][sh + 1][sw + 0] * filter2[fil][15][1][0];
                            accd162 =   accc163 + accc164;
                            acce161 =       accd161 + accd162;
                            accc165 = ifmaps3[15][sh + 1][sw + 1] * filter2[fil][15][1][1];
                            accc166 = ifmaps3[15][sh + 1][sw + 2] * filter2[fil][15][1][2];
                            accd163 =   accc165 + accc166;
                            accc167 = ifmaps3[15][sh + 2][sw + 0] * filter2[fil][15][2][0];
                            accc168 = ifmaps3[15][sh + 2][sw + 1] * filter2[fil][15][2][1];
                            accd164 =   accc167 + accc168;
                            acce162 =       accd163 + accd164;
                            accc169 = ifmaps3[15][sh + 2][sw + 2] * filter2[fil][15][2][2];
                            acc16 =  acce161 + acce162 + accc169;

                            ac8 = acc15 + acc16;
                            a4 = ac7 + ac8;
                            a_c2 = a3 + a4;

                            // a1 + a2 + a3 + a4;   // a_c1 + a_c2;
                            //acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + + acc10  + acc11  + acc12  + acc13  + acc14  + acc15  + acc16;
                            ofmaps[ofm_idx] += a_c1 + a_c2;
                            ofm_idx += dst_h * dst_w;


                        }   // for fil
                    }   // for sw
                }   // for sh
            }   // for chan

            int icp_ofm = 0;
            for(; icp_ofm < dst_c*dst_h*dst_w; icp_ofm++){
                #ifdef SDSOC
                #pragma HLS PIPELINE
                #endif
                C[icp_ofm] = ofmaps[icp_ofm];
            }
            // actor机制
            if(icp_ofm == dst_c*dst_h*dst_w){
                for(;icp_ofm < ofmelmt;){
                    #ifdef SDSOC
                    #pragma HLS PIPELINE
                    #endif
                    C[icp_ofm] = 0;
                    icp_ofm++;
                }
                //std::cout << "actor on, the last icp_ofm = " << icp_ofm - 1 << std::endl;
            }

            break;
        }

        default:
            break;
    }

}
