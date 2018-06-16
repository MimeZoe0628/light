#ifndef LIGHTGBM_BOOSTING_GOSS_H_
#define LIGHTGBM_BOOSTING_GOSS_H_

#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/boosting.h>

#include "score_updater.hpp"
#include "gbdt.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>


#include <LightGBM/metric.h>

namespace LightGBM {

#ifdef TIMETAG
std::chrono::duration<double, std::milli> subset_time;
std::chrono::duration<double, std::milli> re_init_tree_time;
#endif

class GOSS: public GBDT {
public:
  /*!
  * \brief Constructor
  */
  GOSS() : GBDT() {

  }

  ~GOSS() {
    #ifdef TIMETAG
    Log::Info("GOSS::subset costs %f", subset_time * 1e-3);
    Log::Info("GOSS::re_init_tree costs %f", re_init_tree_time * 1e-3);
    #endif
  }
  /*
	GOSS初始化操作：
	  1. 初始化GBDT
	  2. 重置Goss()
  */
  void Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    GBDT::Init(config, train_data, objective_function, training_metrics);
    ResetGoss();
  }

  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override {
    GBDT::ResetTrainingData(train_data, objective_function, training_metrics);
    ResetGoss();
  }

  void ResetConfig(const BoostingConfig* config) override {
    GBDT::ResetConfig(config);
    ResetGoss();
  }

  void ResetGoss() {
	 //检查配置GOSS配置是否有效性
    CHECK(gbdt_config_->top_rate + gbdt_config_->other_rate <= 1.0f);
    CHECK(gbdt_config_->top_rate > 0.0f && gbdt_config_->other_rate > 0.0f);
    if (gbdt_config_->bagging_freq > 0 && gbdt_config_->bagging_fraction != 1.0f) {
      Log::Fatal("cannot use bagging in GOSS");
    }
    Log::Info("using GOSS");

    bag_data_indices_.resize(num_data_);			//bagging 数据索引
    tmp_indices_.resize(num_data_);					//临时数据索引
    tmp_indice_right_.resize(num_data_);			//
    offsets_buf_.resize(num_threads_);
    left_cnts_buf_.resize(num_threads_);
    right_cnts_buf_.resize(num_threads_);
    left_write_pos_buf_.resize(num_threads_);
    right_write_pos_buf_.resize(num_threads_);

    is_use_subset_ = false;
    if (gbdt_config_->top_rate + gbdt_config_->other_rate <= 0.5) {
      auto bag_data_cnt = static_cast<data_size_t>((gbdt_config_->top_rate + gbdt_config_->other_rate) * num_data_);
      tmp_subset_.reset(new Dataset(bag_data_cnt));
      tmp_subset_->CopyFeatureMapperFrom(train_data_);
      is_use_subset_ = true;
    }
    // flag to not bagging first	最开始，标记为不进行bagging
    bag_data_cnt_ = num_data_;
  }
  /*
		/param buffer				存放入样的数据索引
		/param buffer_right			存放未入样的数据索引
		/return cur_left_cnt		入样的样本数
  */
  data_size_t BaggingHelper(Random& cur_rand, data_size_t start, data_size_t cnt, data_size_t* buffer, data_size_t* buffer_right) {
	  //创建大小为当前线程处理的数据量，初始值为0.0的梯度统计量的向量对象
    std::vector<score_t> tmp_gradients(cnt, 0.0f);
    for (data_size_t i = 0; i < cnt; ++i) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
		// 对 一阶导数和二阶导数的乘积取绝对值
        tmp_gradients[i] += std::fabs(gradients_[idx] * hessians_[idx]);
      }
    }
    data_size_t top_k = static_cast<data_size_t>(cnt * gbdt_config_->top_rate);
    data_size_t other_k = static_cast<data_size_t>(cnt * gbdt_config_->other_rate);
    top_k = std::max(1, top_k);
    ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, static_cast<int>(tmp_gradients.size()), top_k);
    score_t threshold = tmp_gradients[top_k - 1];							//按梯度排序后，第top_k个梯度的值
    score_t multiply = static_cast<score_t>(cnt - top_k) / other_k;			//获取抽样比的倒数，用来弥补抽样对分布带来的影响(这么计算是有问题的！)
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_cnt = 0;
    data_size_t big_weight_cnt = 0;											//存放当前收集到的大梯度(top_k)数据的数量
    for (data_size_t i = 0; i < cnt; ++i) {
	  //计算当前实例的一阶导数和二阶导数的乘积取绝对值
      score_t grad = 0.0f;
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {		
        size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
        grad += std::fabs(gradients_[idx] * hessians_[idx]);
      }
	  /*
		存放当前线程中通过GOSS方法得到的数据到buffer中。
		如果：
			当前数据是大梯度数据，则直接存入buffer。
		否则：
			对非大梯度数据进行简单随机抽样。
	  */
      if (grad >= threshold) {
        buffer[cur_left_cnt++] = start + i;
        ++big_weight_cnt;
      } else {//随机抽样
        data_size_t sampled = cur_left_cnt - big_weight_cnt;
        data_size_t rest_need = other_k - sampled;				//接下来需要抽取的样本量
        data_size_t rest_all = (cnt - i) - (top_k - big_weight_cnt);	
        double prob = (rest_need) / static_cast<double>(rest_all);
        if (cur_rand.NextFloat() < prob) {
          buffer[cur_left_cnt++] = start + i;
          for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
            size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
			//更新入样的非大梯度数据的梯度统计信息
            gradients_[idx] *= multiply;
            hessians_[idx] *= multiply;
          }
        } else {
          buffer_right[cur_right_cnt++] = start + i;
        }
      }
    }// endfor
    return cur_left_cnt;
  }

  void Bagging(int iter) override {
    bag_data_cnt_ = num_data_;
    // not subsample for first iterations	头一轮迭代不进行抽样
    if (iter < static_cast<int>(1.0f / gbdt_config_->learning_rate)) { return; }

	//单个线程处理的最小数据条目数
    const data_size_t min_inner_size = 100;
    data_size_t inner_size = (num_data_ + num_threads_ - 1) / num_threads_;			//为什么这么算？？？
    if (inner_size < min_inner_size) { inner_size = min_inner_size; }
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < num_threads_; ++i) {
      OMP_LOOP_EX_BEGIN();
      left_cnts_buf_[i] = 0;
      right_cnts_buf_[i] = 0;
      data_size_t cur_start = i * inner_size;			//当前线程处理的数据的起始偏移量
      if (cur_start > num_data_) { continue; }
      data_size_t cur_cnt = inner_size;
      if (cur_start + cur_cnt > num_data_) { cur_cnt = num_data_ - cur_start; }			//当前线程中处理的数据数量
      Random cur_rand(gbdt_config_->bagging_seed + iter * num_threads_ + i);
	  // 入样的数据数目
      data_size_t cur_left_count = BaggingHelper(cur_rand, cur_start, cur_cnt,
                                                 tmp_indices_.data() + cur_start, tmp_indice_right_.data() + cur_start);
      offsets_buf_[i] = cur_start;						//存放每个线程处理的数据的其起始偏移量（相对于总体数据的偏移量）
      left_cnts_buf_[i] = cur_left_count;				//存放当前线程入样的数据数量
      right_cnts_buf_[i] = cur_cnt - cur_left_count;	//存放当前线程未入样的数据数量
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    data_size_t left_cnt = 0;
    left_write_pos_buf_[0] = 0;
    right_write_pos_buf_[0] = 0;
	/*
		left_write_pos_buf_：用来存储0-n个线程中样本数据写入缓冲区的起始位置
		right_write_pos_buf_：用来存储0-n个线程中非样本数据写入缓冲区的起始位置
	*/
    for (int i = 1; i < num_threads_; ++i) {
      left_write_pos_buf_[i] = left_write_pos_buf_[i - 1] + left_cnts_buf_[i - 1];
      right_write_pos_buf_[i] = right_write_pos_buf_[i - 1] + right_cnts_buf_[i - 1];
    }
	//所有线程抽取的样本数量的和
    left_cnt = left_write_pos_buf_[num_threads_ - 1] + left_cnts_buf_[num_threads_ - 1];

    #pragma omp parallel for schedule(static, 1)
	//将tmp_indices_中的样本数据放到bag_data_indices_的左侧，非样本数据放到bag_data_indices_的右侧
    for (int i = 0; i < num_threads_; ++i) {
      OMP_LOOP_EX_BEGIN();
      if (left_cnts_buf_[i] > 0) {
        std::memcpy(bag_data_indices_.data() + left_write_pos_buf_[i],
                    tmp_indices_.data() + offsets_buf_[i], left_cnts_buf_[i] * sizeof(data_size_t));
      }
      if (right_cnts_buf_[i] > 0) {
        std::memcpy(bag_data_indices_.data() + left_cnt + right_write_pos_buf_[i],
                    tmp_indice_right_.data() + offsets_buf_[i], right_cnts_buf_[i] * sizeof(data_size_t));
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    bag_data_cnt_ = left_cnt;
    // set bagging data to tree learner
    if (!is_use_subset_) {
      tree_learner_->SetBaggingData(bag_data_indices_.data(), bag_data_cnt_);
    } else {
      // get subset  ？？？？
      #ifdef TIMETAG
      auto start_time = std::chrono::steady_clock::now();
      #endif
      tmp_subset_->ReSize(bag_data_cnt_);
      tmp_subset_->CopySubset(train_data_, bag_data_indices_.data(), bag_data_cnt_, false);
      #ifdef TIMETAG
      subset_time += std::chrono::steady_clock::now() - start_time;
      #endif
      #ifdef TIMETAG
      start_time = std::chrono::steady_clock::now();
      #endif
      tree_learner_->ResetTrainingData(tmp_subset_.get());
      #ifdef TIMETAG
      re_init_tree_time += std::chrono::steady_clock::now() - start_time;
      #endif
    }
  }

  /*!
  * \brief Get Type name of this boosting object
  */
  const char* SubModelName() const override { return "tree"; }

private:
  std::vector<data_size_t> tmp_indice_right_;
};


}  // namespace LightGBM
#endif   // LIGHTGBM_BOOSTING_GOSS_H_
