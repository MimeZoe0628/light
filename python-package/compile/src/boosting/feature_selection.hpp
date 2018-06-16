#ifndef LIGHTGBM_FEATURE_SELECTION_H_
#define LIGHTGBM_FEATURE_SELECTION_H_

#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/random.h>
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
#include <math.h> 

namespace LightGBM {

/*!
	动态抽样
*/
class FeatureSelection  {
public:
	
	FeatureSelection(){
	}

	~FeatureSelection() {

	}
	void Init(double feature_fraction,int random_seed) {
		/*
			初始化特征抽样比，用来计算需要选择的特征数
		*/
		feature_fraction_ = feature_fraction;
		total_select_num_ = 0;
		random_ = Random(random_seed);
	}
	/*
		对特征进行均匀随机抽样（该函数在第一个迭代时使用）
		该函数还初始化feature_select_num_
	*/
	std::vector<int> RandomSelection(std::vector<int> valid_feature_indices) {
		//如果不进行特征选择，返回空的向量对象，这是由于学习器不会使用该函数的返回值。
		if (feature_fraction_ >= 1)
			return std::vector<int>();

		//计算需要选择的特征数
		int used_feature_cnt = static_cast<int>(valid_feature_indices.size()*feature_fraction_);
		used_feature_cnt = std::max(used_feature_cnt, 1);

		//均匀随机抽样
		std::vector<int> sampled_indices = random_.Sample(static_cast<int>(valid_feature_indices.size()), used_feature_cnt);

		//初始化feature_select_num_
		if (feature_select_num_.size() == 0) {
			for (int i = 0; i < valid_feature_indices.size(); i++) {
				feature_select_num_.push_back(0);
			}
		}
		//更新特征被选择的次数
		UpdateFeatureSelectNum(&sampled_indices);
		return sampled_indices;
	}
	
	/*!
	*	\brief	简化的UCB特征选择逻辑，根据特征的平均收益来计算特征分数。并返回根据特征分数选择的特征索引
	*/
	std::vector<int> NaiveUCB2(std::vector<double> *gain_importance, std::vector<int> *valid_feature_indices) {
		//如果不进行特征选择，返回空的向量对象，这是由于学习器不会使用该函数的返回值。
		if (feature_fraction_ >= 1)
			return std::vector<int>();

		std::vector<double> scores;
		double max_value = 0;
		std::map<int, int>::iterator iter;
		for (int i = 0; i < valid_feature_indices->size(); i++) {
			// 特征未被选择过，收益分数为0
			if (feature_select_num_[i] == 0)
			{
				scores.push_back(0);
				continue;
			}

			// 计算平均收益分数
			double avg = gain_importance->at(valid_feature_indices->at(i)) / feature_select_num_.at(i);
			scores.push_back(avg);
			// 找出最大平均收益分数，用于归一化
			if (max_value < avg) { 
				max_value = avg;
			}
		}

		// 提前计算特征未被选择情况下的c常量，以防重复的乘法运算
		double constant = sqrt(2 * log(total_select_num_) / 1);
		for (int i = 0; i < valid_feature_indices->size(); i++) {
			double bonus;
			double c;
			double score;
			if (feature_select_num_.at(i) == 0) {
				bonus = 0;
				c = constant;
			}
			else {
				bonus = scores[i] / max_value;
				c = sqrt(2 * log(total_select_num_) / feature_select_num_.at(i));
			}
			score = bonus + c;
			scores[i] = score;
		}

		/*
			特征选择逻辑
		*/
		data_size_t used_feature_cnt = static_cast<int>(valid_feature_indices->size()*feature_fraction_);
		used_feature_cnt = std::max(used_feature_cnt, 1);

		std::vector<double> valid_feature_importance;
		std::vector<double> ordered_score;

		for (auto item : scores) {
			ordered_score.push_back(item);
		}
		// score与valid_feature_indices_元素对应，找到score为前used_feature_cnt个的元素的索引
		std::sort(ordered_score.begin(), ordered_score.end());
		data_size_t threshold_idx = ordered_score.size() - used_feature_cnt;
		double threshold = ordered_score.at(threshold_idx);
		std::vector<int> sampled_indices;					//用来保存选择的特征

		int same_tail_value_num = 0;				//需要被选择的分数与阈值相同的实例数目
		//从阈值所在位置向两头查找与阈值取值相同的实例数目
		//向右侧查找
		for (data_size_t i = threshold_idx; i < ordered_score.size(); i++)
		{
			if (ordered_score.at(i) == threshold) {
				same_tail_value_num++;
			}
			else {
				break;
			}
		}
			
		data_size_t big_threshold_cnt = used_feature_cnt - same_tail_value_num;
		data_size_t big_threshold_sampled = 0;
		for (int i = 0; i < scores.size(); i++)
		{
			// 抽样完毕，即使跳出循环
			if (sampled_indices.size() == used_feature_cnt)
				break;

			// 大于阈值的入样
			if (scores.at(i) > threshold) {
				sampled_indices.push_back(i);
				big_threshold_sampled++;
			}
			/*
				等于阈值时(并非所有等于阈值的实例都入样)：进行特征抽样
			*/
			else if (scores.at(i) == threshold) {
				data_size_t sampled = sampled_indices.size() - big_threshold_sampled;	//已经入样的取值等于阈值的样本数
				data_size_t rest_need = same_tail_value_num - sampled;					//接下来需要抽取的样本量
				data_size_t rest_all = scores.size() - (big_threshold_cnt - big_threshold_sampled);

				double prob = (rest_need) / static_cast<double>(rest_all);
				if (random_.NextFloat() < prob) {
					sampled_indices.push_back(i);
				}
			}
			else {
				continue;
			}
		}
		
		//更新特征被选择的次数
		UpdateFeatureSelectNum(&sampled_indices);
		return sampled_indices;
	}
	// 更新特征选择次数。参数：上一次选择的特征
	void UpdateFeatureSelectNum(std::vector<int> *feature_selected) {
		for (int i = 0; i < feature_selected->size(); i++) {
			data_size_t feat_idx = feature_selected->at(i);
			feature_select_num_[feat_idx] += 1;
		}
		total_select_num_ += feature_selected->size();
	}

	
private:
	std::vector<data_size_t> feature_select_num_;									//特征被选择的次数，有valid_feature_indices对应
	data_size_t total_select_num_;											//总的选择次数
	double feature_fraction_;												//特征抽样比
	Random random_;														
	data_size_t bound_;														//正确分类的概率
	std::vector<std::pair<int, double>> gradients_need_modified_;			//需要更新的梯度<梯度索引,原始值>
	std::vector<std::pair<int, double>> hessians_need_modified_;

};
}  // namespace LightGBM

#endif