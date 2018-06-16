#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/boosting.h>

#include "score_updater.hpp"
#include "gbdt.h"
#include "feature_selection.hpp"
 
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <LightGBM/metric.h>
#include <math.h> 

namespace LightGBM {
#ifdef TIMETAG
	std::chrono::duration<double, std::milli> boosting_time;
	std::chrono::duration<double, std::milli> train_score_time;
	std::chrono::duration<double, std::milli> out_of_bag_score_time;
	std::chrono::duration<double, std::milli> valid_score_time;
	std::chrono::duration<double, std::milli> metric_time;
	std::chrono::duration<double, std::milli> bagging_time;
	std::chrono::duration<double, std::milli> tree_time;
#endif // TIMETAG

/*!
	��̬����
*/
class AdaptiveSampling : public GBDT {
public:
	/*!
	* \brief Constructor
		��Ҫ���õĲ����Լ�Ĭ��ֵ��
		error=0.05
		confidence=0.9
	*/
	AdaptiveSampling() : GBDT() {
		left_write_start_pos_ = 0;
		right_write_start_pos_ = 0;
	}

	~AdaptiveSampling() {
#ifdef TIMETAG
		Log::Info("AdaptiveSampling::subset costs %f", subset_time * 1e-3);
		Log::Info("AdaptiveSampling::re_init_tree costs %f", re_init_tree_time * 1e-3);
#endif
	}
	
	bool IsTerminateSampling(const int num_sample, const double bound) {
		bool result = num_sample >= bound || (num_sample == num_data_);
		Log::Info("adaptive sampling terminate bound: %d >= %f, total: %d", bag_data_cnt_, bound, num_data_);
		return result;
	}
	/*
	��̬�����㷨��ֹ�����߽�
	/param error		���
	/param confidence	���Ŷȣ�1-delta��
	/reference			Sampling Adaptively Using the Massart Inequality for Scalable Learning.2013
	*/
	data_size_t GetBound(const double error, const double confidence, const double prob) {
		double delta = 1 - confidence;
		double bound = 2 * std::log(2 / delta) / std::pow(error, 2)*(0.25 - std::pow(std::abs(prob - 0.5) - 2 / 3 * error, 2));
		return static_cast<data_size_t>(bound);
	}
	void Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
		const std::vector<const Metric*>& training_metrics) override {
		GBDT::Init(config, train_data, objective_function, training_metrics);
		error_ = config->error;
		confidence_ = config->confidence;
		sub_feature_method_ = config->sub_feature_method;
		bound_ = GetBound(error_, confidence_, 0.5);
		ResetAdaptiveSampling();
	}

	void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
		const std::vector<const Metric*>& training_metrics) override {
		GBDT::ResetTrainingData(train_data, objective_function, training_metrics);
		ResetAdaptiveSampling();
	}

	void ResetConfig(const BoostingConfig* config) override {
		GBDT::ResetConfig(config);
		ResetAdaptiveSampling();
	}
	/*
		����AdaptiveSampling
	*/
	void ResetAdaptiveSampling() {
		CHECK(gbdt_config_->error > 0.0f && gbdt_config_->error < 1.0f);
		CHECK(gbdt_config_->confidence > 0.0f && gbdt_config_->confidence < 1.0f);
		if (gbdt_config_->bagging_freq > 0 && gbdt_config_->bagging_fraction != 1.0f) {
			Log::Fatal("Adaptive Sampling�в���ʹ��bagging");
		}
		
		Log::Info("���� Adaptive Sampling");

		bag_data_indices_.resize(num_data_);
		tmp_indices_.resize(num_data_);
		tmp_indice_right_.resize(num_data_);
		offsets_buf_.resize(num_threads_);
		left_cnts_buf_.resize(num_threads_);
		right_cnts_buf_.resize(num_threads_);
		left_write_pos_buf_.resize(num_threads_);
		right_write_pos_buf_.resize(num_threads_);

		////////////
		left_write_start_pos_ = 0;
		right_write_start_pos_ = 0;
		gradients_need_modified_.clear();
		hessians_need_modified_.clear();
		///////////

		// flag to not bagging first
		bag_data_cnt_ = num_data_;
		is_use_subset_ = true;
		auto bag_data_cnt = bag_data_cnt_;
		tmp_subset_.reset(new Dataset(bag_data_cnt));
		tmp_subset_->CopyFeatureMapperFrom(train_data_);
		
	}

	/*
		��̬�������������л��Ż��ĺ���
	/param cnt					��ǰ�̴߳����������
	/param buffer				�����������������
	/param buffer_right			���δ��������������
	/return cur_left_cnt		������������
	*/
	data_size_t BaggingHelper(Random& cur_rand, data_size_t start, data_size_t cnt, data_size_t* buffer, data_size_t* buffer_right) {

		double dft_sampling_base_rate = (bound_ + log2(bound_)) /num_data_;									//Ĭ�ϳ�����
		data_size_t top_k = static_cast<data_size_t>(ceil(cnt * dft_sampling_base_rate));
		top_k = std::max(1, top_k);
		data_size_t cur_left_cnt = 0;
		data_size_t cur_right_cnt = 0;

		for (data_size_t i = 0; i < cnt; ++i) {
			/*score_t grad = 0.0f;
			for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
				size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
				grad += std::fabs(gradients_[idx] * hessians_[idx]);
			}*/
			/*
			�������������ǰ�߳��е��������ݴ�ŵ�buffer�С�
			*/
			data_size_t sampled = cur_left_cnt;									//�Ѿ���ȡ��������
			data_size_t rest_need = top_k - sampled;							//��������Ҫ��ȡ��������
			data_size_t rest_all = cnt - i;										//��δ��������������
			double prob = (rest_need) / static_cast<double>(rest_all);
			if (cur_rand.NextFloat() < prob) {
				buffer[cur_left_cnt++] = start + i;
			}
			else {
				buffer_right[cur_right_cnt++] = start + i;
			}
		}

		return cur_left_cnt;
	}
	/*
		������һ����������
	*/
	bool TrainOneIter(const score_t* gradients, const score_t* hessians) override {
		auto init_score = BoostFromAverage();

		// boosting first
		if (gradients == nullptr || hessians == nullptr) {

#ifdef TIMETAG
			auto start_time = std::chrono::steady_clock::now();
#endif
			// ִ�������������һ�׵����Ͷ��׵���
			Boosting();
			gradients = gradients_.data();
			hessians = hessians_.data();

#ifdef TIMETAG
			boosting_time += std::chrono::steady_clock::now() - start_time;
#endif
		}

#ifdef TIMETAG
		auto start_time = std::chrono::steady_clock::now();
#endif
		bool should_continue = false;
		// ��̬��������ʵ��
		while (true) {
			// bagging logic
			Bagging(iter_);
#ifdef TIMETAG
			bagging_time += std::chrono::steady_clock::now() - start_time;
#endif

			for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {

#ifdef TIMETAG
				start_time = std::chrono::steady_clock::now();
#endif

				std::unique_ptr<Tree> new_tree(new Tree(2));
				if (class_need_train_[cur_tree_id]) {
					size_t bias = static_cast<size_t>(cur_tree_id)* num_data_;
					auto grad = gradients + bias;
					auto hess = hessians + bias;

					// need to copy gradients for bagging subset. Ϊbagging�Ӽ������ݶ�ͳ������
					// �ݶ������С��ԭʼ���ݴ�Сһ�»�����bagging������ݴ�Сһ��???----�����Ż�Ϊ����ѭ����ֵ
					if (is_use_subset_ && bag_data_cnt_ < num_data_) {
						for (int i = 0; i < bag_data_cnt_; ++i) {
							gradients_[bias + i] = grad[bag_data_indices_[i]];
							hessians_[bias + i] = hess[bag_data_indices_[i]];
						}
						grad = gradients_.data() + bias;
						hess = hessians_.data() + bias;
					}
					//////////////////////---------------��������-----------------//////////////////////
					tree_learner_->SetFeatureSamplingByImportance(true);
					if (gbdt_config_->sub_feature_method == "split") {
						if (iter_ != 0) {
							std::vector<double> feature_importances = FeatureImportance(iter_ - 1, 0);
							tree_learner_->FeatureSamplingByImportance(feature_importances);
						}
					}
					else if (gbdt_config_->sub_feature_method == "gain") {
						if (iter_ != 0) {
							std::vector<double> feature_importances = FeatureImportance(iter_ - 1, 1);
							tree_learner_->FeatureSamplingByImportance(feature_importances);
						}
					}
					else if (gbdt_config_->sub_feature_method == "ucb") {
						if (iter_ == 0) {
							tree_learner_->SetFeatureSampledIndices(feature_selection_->RandomSelection(tree_learner_->GetValidFeatureIndices()));
						}
						else {
							std::vector<double> gain_importance = FeatureImportance(iter_ + 1, 1);
							std::vector<int> valid_feature_indices = tree_learner_->GetValidFeatureIndices();
							tree_learner_->SetFeatureSampledIndices(feature_selection_->NaiveUCB2(&gain_importance, &valid_feature_indices));
						}// end if
					}
					else {
						tree_learner_->SetFeatureSamplingByImportance(false);
					}

					// ʹ���µ�ͳ�����ݺ�ѵ�����ݽ���ѵ����������������
					new_tree.reset(tree_learner_->Train(grad, hess, is_constant_hessian_));
				}

#ifdef TIMETAG
				tree_time += std::chrono::steady_clock::now() - start_time;
#endif

				if (new_tree->num_leaves() > 1) {
					should_continue = true;
					// shrinkage by learning rate
					new_tree->Shrinkage(shrinkage_rate_);
					// update score
					UpdateScore(new_tree.get(), cur_tree_id);
					if (std::fabs(init_score) > kEpsilon) {
						new_tree->AddBias(init_score);
					}
				}
				else {
					// only add default score one-time
					if (!class_need_train_[cur_tree_id] && models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
						auto output = class_default_output_[cur_tree_id];
						new_tree->AsConstantTree(output);
						// updates scores
						train_score_updater_->AddScore(output, cur_tree_id);
						for (auto& score_updater : valid_score_updater_) {
							score_updater->AddScore(output, cur_tree_id);
						}
					}
				}
				// add model
				models_.push_back(std::move(new_tree));
			}

			bool terminate = false;
			double prob = 0;
			/////////////////////////////	��ȡ��ǰģ�Ͷ���������ķ��������
			for (auto& sub_metric : training_metrics_) {
				auto name = sub_metric->GetName();
				auto scores = EvalOneMetric(sub_metric, train_score_updater_->score());
				for (size_t k = 0; k < name.size(); ++k) {
					if (name[k] != std::string("binary_error")) { continue; }
					std::stringstream tmp_buf;
					tmp_buf << " ��ǰ����:" << iter_
						<< ", ѵ���� " << name[k]
						<< " : " << scores[k];

					bound_ = GetBound(error_, confidence_, scores[k]);
					terminate = IsTerminateSampling(bag_data_cnt_, bound_);
					prob = 1 - scores[k];
					Log::Info(tmp_buf.str().c_str());
				}
			}
			// �ж��Ƿ���Ҫ��������
			if (terminate && prob > 0.5) {
				last_reward_ = 2 * (1 - prob);
				break;
			}
			else {
				for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
					models_.pop_back();
				}
			}
		}// end for
		
		 /////////////////////////////
		if (!should_continue) {
			Log::Warning("Stopped training because there are no more leaves that meet the split requirements.");
			for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
				models_.pop_back();
			}
			return true;
		}
		
		++iter_;
		//��һ�ֵ������������붯̬����������صĳ�Ա��������
		left_write_start_pos_ = 0;
		right_write_start_pos_ = 0;
		gradients_need_modified_.clear();
		hessians_need_modified_.clear();
		return false;
	}
	void Bagging(int iter) override {
		bag_data_cnt_ = num_data_ - left_write_start_pos_;
		// not subsample for first iterations	��һ������������
		//if (iter < static_cast<int>(1.0f / gbdt_config_->learning_rate)) { return; }

		//�����̴߳������С������Ŀ��
		const data_size_t min_inner_size = 100;
		data_size_t inner_size = (num_data_ - left_write_start_pos_ + num_threads_ - 1) / num_threads_;			//Ϊʲô��ô�㣿����
		if (inner_size < min_inner_size) { inner_size = min_inner_size; }
		OMP_INIT_EX();
#pragma omp parallel for schedule(static, 1)
		for (int i = 0; i < num_threads_; ++i) {
			OMP_LOOP_EX_BEGIN();
			left_cnts_buf_[i] = 0;
			right_cnts_buf_[i] = 0;
			data_size_t cur_start = i * inner_size + left_write_start_pos_;	//��ǰ�̴߳�������ݵ���ʼλ��
			if (cur_start > num_data_) { continue; }
			data_size_t cur_cnt = inner_size;								//��ǰ�̴߳������������С
			if (cur_start + cur_cnt > num_data_) { cur_cnt = num_data_ - cur_start; }
			Random cur_rand(gbdt_config_->bagging_seed + iter * num_threads_ + i);
			// ������������Ŀ
			data_size_t cur_left_count = BaggingHelper(cur_rand, cur_start, cur_cnt,
				tmp_indices_.data() + cur_start, tmp_indice_right_.data() + cur_start);
			offsets_buf_[i] = cur_start;								//���ÿ���̴߳�������ݵ�����ʼƫ����
			left_cnts_buf_[i] = cur_left_count;							//��ŵ�ǰ�߳���������������
			right_cnts_buf_[i] = cur_cnt - cur_left_count;				//��ŵ�ǰ�߳�δ��������������
			OMP_LOOP_EX_END();
		}
		OMP_THROW_EX();
		data_size_t left_cnt = 0;
		left_write_pos_buf_[0] = left_write_start_pos_;					//������������������Ҫд�����ʼλ��
		right_write_pos_buf_[0] = right_write_start_pos_;				//����������������Ҫд�����ʼλ��
		for (int i = 1; i < num_threads_; ++i) {
			left_write_pos_buf_[i] = left_write_pos_buf_[i - 1] + left_cnts_buf_[i - 1];
			right_write_pos_buf_[i] = right_write_pos_buf_[i - 1] + right_cnts_buf_[i - 1];
		}

		//�����̳߳�ȡ�����������ĺ�
		left_cnt = left_write_pos_buf_[num_threads_ - 1] + left_cnts_buf_[num_threads_ - 1] - left_write_start_pos_;

#pragma omp parallel for schedule(static, 1)
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

		left_write_start_pos_ += left_cnt;
		right_write_start_pos_ += left_cnt;

		bag_data_cnt_ = left_write_start_pos_;									//��ǰ�Ѿ�������������С

																				//���������Ҫ�޸ĵ��ݶ�����
		//for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
		//	for (int cur_data_idx = (bag_data_cnt_ - left_cnt); cur_data_idx < bag_data_cnt_; ++cur_data_idx) {
		//		size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + bag_data_indices_[cur_data_idx];
		//		//������˴γ�ȡ��������ص���Ҫ�޸ĵ��ݶ����ݵ�ԭʼֵ
		//		gradients_need_modified_.emplace_back(idx, gradients_[idx]);
		//		hessians_need_modified_.emplace_back(idx, hessians_[idx]);
		//	}
		//}
		////���µ�ĿǰΪֹ������Ҫ�޸ĵ��ݶ�����
		//score_t multiply = static_cast<score_t>(num_data_) / bag_data_cnt_;		//��ȡ�����ȵĵ����������ֲ������Էֲ�������Ӱ��
		//for (int i = 0; i < gradients_need_modified_.size(); i++) {
		//	gradients_[gradients_need_modified_[i].first] = gradients_need_modified_[i].second * multiply;
		//	hessians_[hessians_need_modified_[i].first] = hessians_need_modified_[i].second * multiply;
		//}

		// set bagging data to tree learner
		if (!is_use_subset_) {
			Log::Info("bagging data num: d%", bag_data_cnt_);
			tree_learner_->SetBaggingData(bag_data_indices_.data(), bag_data_cnt_);
		}
		else {
			// get subset
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
	* \brief ��ȡ��ǰboosting���������
	*/
	const char* SubModelName() const override { return "tree"; }

private:
	std::vector<data_size_t> tmp_indice_right_;
	double error_;
	double confidence_;
	std::string sub_feature_method_;										//������������
	data_size_t bound_;														//��ȷ����ĸ���
	data_size_t left_write_start_pos_;
	data_size_t right_write_start_pos_;
	double last_reward_;													//��һ�εĻر�����������ѡ��
	std::vector<std::pair<int, double>> gradients_need_modified_;			//��Ҫ���µ��ݶ�<�ݶ�����,ԭʼֵ>
	std::vector<std::pair<int, double>> hessians_need_modified_;

};
}  // namespace LightGBM

