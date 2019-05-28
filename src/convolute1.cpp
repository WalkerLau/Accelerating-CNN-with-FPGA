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

#include "datatype.h"
#include "convolute1.h"

void convolute1(const data_1 A[3*228*228], const data_2 B[48*3*9*9], data_1 C[48*55*55], const int m, const int k){

    const int ifmelmt = 3*228*228;
    const int filelmt = 48*3*9*9;
    const int ofmelmt = 48*55*55;

    int fac_para_1 = 1; 
    int fac_para = fac_para_1;           // 第一层fac_para定制为1.
    const int src_c = 3;
    const int src_h = 228;
    const int src_w = 228;
    const int stride = 4;
    const int filter_size = 3*9*9;
    const int filter_h = 9;
    const int filter_w = 9;
    const int dst_c = 48;
    const int dst_h = 55;
    const int dst_w = 55;
    const int end_h = src_h - filter_h + 1;
    const int end_w = src_w - filter_w + 1;

    data_1 ifmaps1[1][228][228];            // 存第一层ifmaps。
    data_2 filter1[48][1][9][9];            // 存第一层filter。
    data_1 ofmaps[48*55*55];
    #ifdef SDSOC
        #pragma HLS array_partition variable=ifmaps1 complete dim=1
		#pragma HLS array_partition variable=ifmaps1 cyclic factor=2 dim=2
		#pragma HLS array_partition variable=ifmaps1 cyclic factor=2 dim=3
        #pragma HLS array_partition variable=filter1 complete dim=2
        #pragma HLS array_partition variable=filter1 complete dim=3
        #pragma HLS array_partition variable=filter1 complete dim=4
    #endif

    data_1 actor_1;   data_2 actor_2;   // actor机制。

    data_3 a_c1;      //data_3 a_c2;

    data_3 a1;        data_3 a2;        //data_3 a3;         data_3 a4;

    data_3 ac1;       data_3 ac2;       data_3 ac3;        data_3 ac4;       //data_3 ac5;       data_3 ac6;       data_3 ac7;      data_3 ac8;

    data_3 acc1;      data_3 acc2;      data_3 acc3;       data_3 acc4;      data_3 acc5;      data_3 acc6;      data_3 acc7;     data_3 acc8;     data_3 acc9;     //data_3 acc10;     data_3 acc11;    data_3 acc12;    data_3 acc13;    data_3 acc14;    data_3 acc15;    data_3 acc16;

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
                            ifmaps1[cidx][hidx][widx] = A[icp_ifm];
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
                                filter1[fil][cidx][hidx][widx] = B[icp_fil];
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
                            accc011 = ifmaps1[0][sh + 0][sw + 0] * filter1[fil][0][0][0];
                            accc012 = ifmaps1[0][sh + 0][sw + 1] * filter1[fil][0][0][1];
                            accd011 =   accc011 + accc012;
                            accc013 = ifmaps1[0][sh + 0][sw + 2] * filter1[fil][0][0][2];
                            accc014 = ifmaps1[0][sh + 0][sw + 3] * filter1[fil][0][0][3];
                            accd012 =   accc013 + accc014;
                            acce011 =       accd011 + accd012;
                            accc015 = ifmaps1[0][sh + 0][sw + 4] * filter1[fil][0][0][4];
                            accc016 = ifmaps1[0][sh + 0][sw + 5] * filter1[fil][0][0][5];
                            accd013 =   accc015 + accc016;
                            accc017 = ifmaps1[0][sh + 0][sw + 6] * filter1[fil][0][0][6];
                            accc018 = ifmaps1[0][sh + 0][sw + 7] * filter1[fil][0][0][7];
                            accd014 =   accc017 + accc018;
                            acce012 =       accd013 + accd014;
                            accc019 = ifmaps1[0][sh + 0][sw + 8] * filter1[fil][0][0][8];
                            acc1 =  acce011 + acce012 + accc019;

                            accc021 = ifmaps1[0][sh + 1][sw + 0] * filter1[fil][0][1][0];
                            accc022 = ifmaps1[0][sh + 1][sw + 1] * filter1[fil][0][1][1];
                            accd021 =   accc021 + accc022;
                            accc023 = ifmaps1[0][sh + 1][sw + 2] * filter1[fil][0][1][2];
                            accc024 = ifmaps1[0][sh + 1][sw + 3] * filter1[fil][0][1][3];
                            accd022 =   accc023 + accc024;
                            acce021 =       accd021 + accd022;
                            accc025 = ifmaps1[0][sh + 1][sw + 4] * filter1[fil][0][1][4];
                            accc026 = ifmaps1[0][sh + 1][sw + 5] * filter1[fil][0][1][5];
                            accd023 =   accc025 + accc026;
                            accc027 = ifmaps1[0][sh + 1][sw + 6] * filter1[fil][0][1][6];
                            accc028 = ifmaps1[0][sh + 1][sw + 7] * filter1[fil][0][1][7];
                            accd024 =   accc027 + accc028;
                            acce022 =       accd023 + accd024;
                            accc029 = ifmaps1[0][sh + 1][sw + 8] * filter1[fil][0][1][8];
                            acc2 =  acce021 + acce022 + accc029;

                            ac1 = acc1 + acc2;

                            accc031 = ifmaps1[0][sh + 2][sw + 0] * filter1[fil][0][2][0];
                            accc032 = ifmaps1[0][sh + 2][sw + 1] * filter1[fil][0][2][1];
                            accd031 =   accc031 + accc032;
                            accc033 = ifmaps1[0][sh + 2][sw + 2] * filter1[fil][0][2][2];
                            accc034 = ifmaps1[0][sh + 2][sw + 3] * filter1[fil][0][2][3];
                            accd032 =   accc033 + accc034;
                            acce031 =       accd031 + accd032;
                            accc035 = ifmaps1[0][sh + 2][sw + 4] * filter1[fil][0][2][4];
                            accc036 = ifmaps1[0][sh + 2][sw + 5] * filter1[fil][0][2][5];
                            accd033 =   accc035 + accc036;
                            accc037 = ifmaps1[0][sh + 2][sw + 6] * filter1[fil][0][2][6];
                            accc038 = ifmaps1[0][sh + 2][sw + 7] * filter1[fil][0][2][7];
                            accd034 =   accc037 + accc038;
                            acce032 =       accd033 + accd034;
                            accc039 = ifmaps1[0][sh + 2][sw + 8] * filter1[fil][0][2][8];
                            acc3 =  acce031 + acce032 + accc039;

                            accc041 = ifmaps1[0][sh + 3][sw + 0] * filter1[fil][0][3][0];
                            accc042 = ifmaps1[0][sh + 3][sw + 1] * filter1[fil][0][3][1];
                            accd041 =   accc041 + accc042;
                            accc043 = ifmaps1[0][sh + 3][sw + 2] * filter1[fil][0][3][2];
                            accc044 = ifmaps1[0][sh + 3][sw + 3] * filter1[fil][0][3][3];
                            accd042 =   accc043 + accc044;
                            acce041 =       accd041 + accd042;
                            accc045 = ifmaps1[0][sh + 3][sw + 4] * filter1[fil][0][3][4];
                            accc046 = ifmaps1[0][sh + 3][sw + 5] * filter1[fil][0][3][5];
                            accd043 =   accc045 + accc046;
                            accc047 = ifmaps1[0][sh + 3][sw + 6] * filter1[fil][0][3][6];
                            accc048 = ifmaps1[0][sh + 3][sw + 7] * filter1[fil][0][3][7];
                            accd044 =   accc047 + accc048;
                            acce042 =       accd043 + accd044;
                            accc049 = ifmaps1[0][sh + 3][sw + 8] * filter1[fil][0][3][8];
                            acc4 =  acce041 + acce042 + accc049;

                            ac2 = acc3 +acc4;
                            a1 = ac1 + ac2;

                            accc051 = ifmaps1[0][sh + 4][sw + 0] * filter1[fil][0][4][0];
                            accc052 = ifmaps1[0][sh + 4][sw + 1] * filter1[fil][0][4][1];
                            accd051 =   accc051 + accc052;
                            accc053 = ifmaps1[0][sh + 4][sw + 2] * filter1[fil][0][4][2];
                            accc054 = ifmaps1[0][sh + 4][sw + 3] * filter1[fil][0][4][3];
                            accd052 =   accc053 + accc054;
                            acce051 =       accd051 + accd052;
                            accc055 = ifmaps1[0][sh + 4][sw + 4] * filter1[fil][0][4][4];
                            accc056 = ifmaps1[0][sh + 4][sw + 5] * filter1[fil][0][4][5];
                            accd053 =   accc055 + accc056;
                            accc057 = ifmaps1[0][sh + 4][sw + 6] * filter1[fil][0][4][6];
                            accc058 = ifmaps1[0][sh + 4][sw + 7] * filter1[fil][0][4][7];
                            accd054 =   accc057 + accc058;
                            acce052 =       accd053 + accd054;
                            accc059 = ifmaps1[0][sh + 4][sw + 8] * filter1[fil][0][4][8];
                            acc5 =  acce051 + acce052 + accc059;

                            accc061 = ifmaps1[0][sh + 5][sw + 0] * filter1[fil][0][5][0];
                            accc062 = ifmaps1[0][sh + 5][sw + 1] * filter1[fil][0][5][1];
                            accd061 =   accc061 + accc062;
                            accc063 = ifmaps1[0][sh + 5][sw + 2] * filter1[fil][0][5][2];
                            accc064 = ifmaps1[0][sh + 5][sw + 3] * filter1[fil][0][5][3];
                            accd062 =   accc063 + accc064;
                            acce061 =       accd061 + accd062;
                            accc065 = ifmaps1[0][sh + 5][sw + 4] * filter1[fil][0][5][4];
                            accc066 = ifmaps1[0][sh + 5][sw + 5] * filter1[fil][0][5][5];
                            accd063 =   accc065 + accc066;
                            accc067 = ifmaps1[0][sh + 5][sw + 6] * filter1[fil][0][5][6];
                            accc068 = ifmaps1[0][sh + 5][sw + 7] * filter1[fil][0][5][7];
                            accd064 =   accc067 + accc068;
                            acce062 =       accd063 + accd064;
                            accc069 = ifmaps1[0][sh + 5][sw + 8] * filter1[fil][0][5][8];
                            acc6 =  acce061 + acce062 + accc069;

                            ac3 = acc5 + acc6;

                            accc071 = ifmaps1[0][sh + 6][sw + 0] * filter1[fil][0][6][0];
                            accc072 = ifmaps1[0][sh + 6][sw + 1] * filter1[fil][0][6][1];
                            accd071 =   accc071 + accc072;
                            accc073 = ifmaps1[0][sh + 6][sw + 2] * filter1[fil][0][6][2];
                            accc074 = ifmaps1[0][sh + 6][sw + 3] * filter1[fil][0][6][3];
                            accd072 =   accc073 + accc074;
                            acce071 =       accd071 + accd072;
                            accc075 = ifmaps1[0][sh + 6][sw + 4] * filter1[fil][0][6][4];
                            accc076 = ifmaps1[0][sh + 6][sw + 5] * filter1[fil][0][6][5];
                            accd073 =   accc075 + accc076;
                            accc077 = ifmaps1[0][sh + 6][sw + 6] * filter1[fil][0][6][6];
                            accc078 = ifmaps1[0][sh + 6][sw + 7] * filter1[fil][0][6][7];
                            accd074 =   accc077 + accc078;
                            acce072 =       accd073 + accd074;
                            accc079 = ifmaps1[0][sh + 6][sw + 8] * filter1[fil][0][6][8];
                            acc7 =  acce071 + acce072 + accc079;

                            accc081 = ifmaps1[0][sh + 7][sw + 0] * filter1[fil][0][7][0];
                            accc082 = ifmaps1[0][sh + 7][sw + 1] * filter1[fil][0][7][1];
                            accd081 =   accc081 + accc082;
                            accc083 = ifmaps1[0][sh + 7][sw + 2] * filter1[fil][0][7][2];
                            accc084 = ifmaps1[0][sh + 7][sw + 3] * filter1[fil][0][7][3];
                            accd082 =   accc083 + accc084;
                            acce081 =       accd081 + accd082;
                            accc085 = ifmaps1[0][sh + 7][sw + 4] * filter1[fil][0][7][4];
                            accc086 = ifmaps1[0][sh + 7][sw + 5] * filter1[fil][0][7][5];
                            accd083 =   accc085 + accc086;
                            accc087 = ifmaps1[0][sh + 7][sw + 6] * filter1[fil][0][7][6];
                            accc088 = ifmaps1[0][sh + 7][sw + 7] * filter1[fil][0][7][7];
                            accd084 =   accc087 + accc088;
                            acce082 =       accd083 + accd084;
                            accc089 = ifmaps1[0][sh + 7][sw + 8] * filter1[fil][0][7][8];
                            acc8 =  acce081 + acce082 + accc089;

                            ac4 = acc7 + acc8;
                            a2 = ac3 + ac4;
                            a_c1 = a1 + a2;

                            accc091 = ifmaps1[0][sh + 8][sw + 0] * filter1[fil][0][8][0];
                            accc092 = ifmaps1[0][sh + 8][sw + 1] * filter1[fil][0][8][1];
                            accd091 =   accc091 + accc092;
                            accc093 = ifmaps1[0][sh + 8][sw + 2] * filter1[fil][0][8][2];
                            accc094 = ifmaps1[0][sh + 8][sw + 3] * filter1[fil][0][8][3];
                            accd092 =   accc093 + accc094;
                            acce091 =       accd091 + accd092;
                            accc095 = ifmaps1[0][sh + 8][sw + 4] * filter1[fil][0][8][4];
                            accc096 = ifmaps1[0][sh + 8][sw + 5] * filter1[fil][0][8][5];
                            accd093 =   accc095 + accc096;
                            accc097 = ifmaps1[0][sh + 8][sw + 6] * filter1[fil][0][8][6];
                            accc098 = ifmaps1[0][sh + 8][sw + 7] * filter1[fil][0][8][7];
                            accd094 =   accc097 + accc098;
                            acce092 =       accd093 + accd094;
                            accc099 = ifmaps1[0][sh + 8][sw + 8] * filter1[fil][0][8][8];
                            acc9 =  acce091 + acce092 + accc099;

                            // a1 + a2 + a3 + a4;   // a_c1 + a_c2;
                            //acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + + acc10  + acc11  + acc12  + acc13  + acc14  + acc15  + acc16;
                            ofmaps[ofm_idx] += a_c1 + acc9;
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

}
