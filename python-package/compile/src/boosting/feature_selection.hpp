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
	��̬����
*/
class FeatureSelection  {
public:
	
	FeatureSelection(){
	}

	~FeatureSelection() {

	}
	void Init(double feature_fraction,int random_seed) {
		/*
			��ʼ�����������ȣ�����������Ҫѡ���������
		*/
		feature_fraction_ = feature_fraction;
		total_select_num_ = 0;
		random_ = Random(random_seed);
	}
	/*
		���������о�������������ú����ڵ�һ������ʱʹ�ã�
		�ú�������ʼ��feature_select_num_
	*/
	std::vector<int> RandomSelection(std::vector<int> valid_feature_indices) {
		//�������������ѡ�񣬷��ؿյ�����������������ѧϰ������ʹ�øú����ķ���ֵ��
		if (feature_fraction_ >= 1)
			return std::vector<int>();

		//������Ҫѡ���������
		int used_feature_cnt = static_cast<int>(valid_feature_indices.size()*feature_fraction_);
		used_feature_cnt = std::max(used_feature_cnt, 1);

		//�����������
		std::vector<int> sampled_indices = random_.Sample(static_cast<int>(valid_feature_indices.size()), used_feature_cnt);

		//��ʼ��feature_select_num_
		if (feature_select_num_.size() == 0) {
			for (int i = 0; i < valid_feature_indices.size(); i++) {
				feature_select_num_.push_back(0);
			}
		}
		//����������ѡ��Ĵ���
		UpdateFeatureSelectNum(&sampled_indices);
		return sampled_indices;
	}
	
	/*!
	*	\brief	�򻯵�UCB����ѡ���߼�������������ƽ���������������������������ظ�����������ѡ�����������
	*/
	std::vector<int> NaiveUCB2(std::vector<double> *gain_importance, std::vector<int> *valid_feature_indices) {
		//�������������ѡ�񣬷��ؿյ�����������������ѧϰ������ʹ�øú����ķ���ֵ��
		if (feature_fraction_ >= 1)
			return std::vector<int>();

		std::vector<double> scores;
		double max_value = 0;
		std::map<int, int>::iterator iter;
		for (int i = 0; i < valid_feature_indices->size(); i++) {
			// ����δ��ѡ������������Ϊ0
			if (feature_select_num_[i] == 0)
			{
				scores.push_back(0);
				continue;
			}

			// ����ƽ���������
			double avg = gain_importance->at(valid_feature_indices->at(i)) / feature_select_num_.at(i);
			scores.push_back(avg);
			// �ҳ����ƽ��������������ڹ�һ��
			if (max_value < avg) { 
				max_value = avg;
			}
		}

		// ��ǰ��������δ��ѡ������µ�c�������Է��ظ��ĳ˷�����
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
			����ѡ���߼�
		*/
		data_size_t used_feature_cnt = static_cast<int>(valid_feature_indices->size()*feature_fraction_);
		used_feature_cnt = std::max(used_feature_cnt, 1);

		std::vector<double> valid_feature_importance;
		std::vector<double> ordered_score;

		for (auto item : scores) {
			ordered_score.push_back(item);
		}
		// score��valid_feature_indices_Ԫ�ض�Ӧ���ҵ�scoreΪǰused_feature_cnt����Ԫ�ص�����
		std::sort(ordered_score.begin(), ordered_score.end());
		data_size_t threshold_idx = ordered_score.size() - used_feature_cnt;
		double threshold = ordered_score.at(threshold_idx);
		std::vector<int> sampled_indices;					//��������ѡ�������

		int same_tail_value_num = 0;				//��Ҫ��ѡ��ķ�������ֵ��ͬ��ʵ����Ŀ
		//����ֵ����λ������ͷ��������ֵȡֵ��ͬ��ʵ����Ŀ
		//���Ҳ����
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
			// ������ϣ���ʹ����ѭ��
			if (sampled_indices.size() == used_feature_cnt)
				break;

			// ������ֵ������
			if (scores.at(i) > threshold) {
				sampled_indices.push_back(i);
				big_threshold_sampled++;
			}
			/*
				������ֵʱ(�������е�����ֵ��ʵ��������)��������������
			*/
			else if (scores.at(i) == threshold) {
				data_size_t sampled = sampled_indices.size() - big_threshold_sampled;	//�Ѿ�������ȡֵ������ֵ��������
				data_size_t rest_need = same_tail_value_num - sampled;					//��������Ҫ��ȡ��������
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
		
		//����������ѡ��Ĵ���
		UpdateFeatureSelectNum(&sampled_indices);
		return sampled_indices;
	}
	// ��������ѡ���������������һ��ѡ�������
	void UpdateFeatureSelectNum(std::vector<int> *feature_selected) {
		for (int i = 0; i < feature_selected->size(); i++) {
			data_size_t feat_idx = feature_selected->at(i);
			feature_select_num_[feat_idx] += 1;
		}
		total_select_num_ += feature_selected->size();
	}

	
private:
	std::vector<data_size_t> feature_select_num_;									//������ѡ��Ĵ�������valid_feature_indices��Ӧ
	data_size_t total_select_num_;											//�ܵ�ѡ�����
	double feature_fraction_;												//����������
	Random random_;														
	data_size_t bound_;														//��ȷ����ĸ���
	std::vector<std::pair<int, double>> gradients_need_modified_;			//��Ҫ���µ��ݶ�<�ݶ�����,ԭʼֵ>
	std::vector<std::pair<int, double>> hessians_need_modified_;

};
}  // namespace LightGBM

#endif