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
 * The codes are mainly developed by Wanglong Wu(a Ph.D supervised by Prof. Shiguang Shan)
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
 */

#include "common_net.h"

CommonNet::CommonNet() {
  nets_.clear();
  
  input_blobs_.clear();
  output_blobs_.clear();

  input_plugs_.clear();
  output_plugs_.clear();

  params_.clear();
}

CommonNet::~CommonNet() {
  nets_.clear();
  
  input_blobs_.clear();
  output_blobs_.clear();

  input_plugs_.clear();
  output_plugs_.clear();

  params_.clear();
}


std::shared_ptr<Net> CommonNet::Load(FILE* file) {
  
  // Todo: assert file format
  int len;  
  CHECK_EQ(fread(&len, sizeof(int), 1, file), 1);
  char* net_type = new char[len + 1];

  // net type
  CHECK_EQ(fread(net_type, sizeof(char), len, file), len);
  net_type[len] = '\0';
  LOG(INFO) << "Creating " << net_type << " net ...";
  
  //根据所需类型创建网络；并用指向Net类型对象的智能指针net来指向该新建网络类对象。
  std::shared_ptr<Net> net = NetRegistry::CreateNet(net_type);	

  // params
  net->hyper_param()->Load(file);		
  // Todo: name

  net->SetUp();		

  // chg 测---------------------------------------------------------------
  std::cout << "net type:" << net_type << "\nparams_blobs size is : " << net->params().size() << std::endl;

  // chg 测---------------------------------------------------------------
  std::cout << "param loop --- begin " << std::endl;

  // 把项目组model的shape_、data_等参数复制给net的成员params_
  for (int i = 0; i < net->params().size(); ++ i) {		// 循环的次数是net所指对象的params_成员的长度（个数）。
    Blob param(file);	// 把项目组model的特征值参数初始化给Blob类对象param的数据成员shape_和data_。

    LOG(INFO) << net_type << " net blobs[" << i << "]: (" << param.num() << ","
    << param.channels() << "," << param.height() << ","<< param.width() << ")";
		
	// chg 测---------------------------------------------------------------
	std::cout << "net blobs[" << i << "]: " << param.num() << ","
			  << param.channels() << "," << param.height() << "," << param.width() << std::endl;

	net->params(i)->SetData(param);		
  }

  // chg 测---------------------------------------------------------------
  std::cout << "param loop --- done " << std::endl;

  int num_subnet = net->nets().size();
  int num_in = net->input_blobs().size();
  int num_out = net->output_blobs().size();
 
  std::vector<std::shared_ptr<Net> >& nets = net->nets();
  std::vector<Blob>& input_blobs = net->input_blobs();
  std::vector<Blob>& output_blobs = net->output_blobs();
  std::vector<std::vector<Blob*> >& output_plugs = net->output_plugs();
  std::vector<std::vector<Blob*> >& input_plugs = net->input_plugs();
    
  // subnet；从文件加载子网络
  for (int i = 0; i < num_subnet; ++ i) {

	// chg 测：追踪网络的嵌套情况---------------------------------------------------------------
	std::cout << "Subnet load --- begin---as index :" << i << std::endl;

    nets[i] = Load(file);

	// chg 测：追踪网络的嵌套情况---------------------------------------------------------------
	std::cout << "Subnet load --- done ---as index :" << i << std::endl;

    nets[i]->SetFather(net.get());		
  }

  // input and output plugs
  if (num_subnet == 0) {
    for (int i = 0; i < num_in; ++ i) {
      input_plugs[i].push_back(&(input_blobs[i]));	
    }
  }
  else {
    // connection of each subnet
    for (int i = 0; i < num_subnet; ++ i) {
      int input_num = nets[i]->input_blobs().size();
      for (int j = 0; j < input_num; ++ j) {
        int net_idx, blob_idx;
        CHECK_EQ(fread(&net_idx, sizeof(int), 1, file), 1);
        CHECK_EQ(fread(&blob_idx, sizeof(int), 1, file), 1);
        if (net_idx == -1) { // connected to father net
          input_plugs[blob_idx].push_back(nets[i]->input_blobs(j));
        }
        else {
          nets[net_idx]->output_plugs(blob_idx).push_back(
            nets[i]->input_blobs(j));	
        }
      }
    }

    // get output blobs
    for (int i = 0; i < num_out; ++ i) {
      int net_idx, blob_idx;
      CHECK_EQ(fread(&net_idx, sizeof(int), 1, file), 1);
      CHECK_EQ(fread(&blob_idx, sizeof(int), 1, file), 1);
      nets[net_idx]->output_plugs(blob_idx).push_back(
          &(output_blobs[i]));		
    }
    for (int i = 0; i < num_subnet; ++ i) {	
      if (nets[i]->num_output() > 0) {
        LOG(ERROR) << "There are " << nets[i]->num_output() 
          << " output blobs unlinked!";
        exit(0);
      }
    }
  }
  delete []net_type;
  return net;
}

void CommonNet::SetUp() {
  int num_subnet = *(int*)(this->hyper_param()->param("num_subnet"));
  int num_in = *(int*)(this->hyper_param()->param("num_in"));
  int num_out = *(int*)(this->hyper_param()->param("num_out"));
  
  input_blobs_.resize(num_in);
  output_blobs_.resize(num_out);

  input_plugs_.resize(num_in);
  output_plugs_.resize(num_out);
  
  nets_.resize(num_subnet);

  params_.clear();
}

void CommonNet::Execute() {
  LOG(DEBUG) << "Common net executing ...";
  // 1. check input blobs
  for (int i = 0; i < input_blobs_.size(); ++ i) {
    if (input_blobs_[i].data() == nullptr) {
      LOG(INFO) << "Net input haven't been initialized completely!";
      return ;
    }
    // connecting input plugs
    for (std::vector<Blob*>::iterator blob = input_plugs_[i].begin();
        blob != input_plugs_[i].end(); ++ blob) {
      (*blob)->SetData(input_blobs_[i]);
    }
    // release input blobs
    input_blobs_[i].Release();
  }
  LOG(DEBUG) << "Check input blobs done!";
  // 2. execute
  for (int i = 0; i < nets_.size(); ++ i) {
    nets_[i]->Execute();
  }
  /*freopen("debug.txt", "w", stdout);
  for (int i = 0; i < output_blobs_[0].count(); ++i) {
	std::cout << (output_blobs_[0].data().get())[i] << std::endl;
  }*/
  // 3. check output blobs
  for (int i = 0; i < output_blobs_.size(); ++ i) {
    // connecting output plugs
    for (std::vector<Blob*>::iterator blob = output_plugs_[i].begin();
        blob != output_plugs_[i].end(); ++ blob) {
      (*blob)->SetData(output_blobs_[i]);
    }
    // release output blobs
    if (output_plugs_[i].size() != 0) {
      output_blobs_[i].Release();
    }
  }
}

REGISTER_NET_CLASS(Common);
