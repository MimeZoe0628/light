#include "parallel_tree_learner.h"

#include <cstring>

#include <tuple>
#include <vector>

namespace LightGBM {

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::DataParallelTreeLearner(const TreeConfig* tree_config)
  :TREELEARNER_T(tree_config) {
}

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::~DataParallelTreeLearner() {

}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner					初始化SerialTreeLearner
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  // Get local rank and global machine size			获取本地rank和全局机器大小
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();
  // allocate buffer for communication				为通信分配缓冲区(即训练数据占据的所有bin的大小)
  size_t buffer_size = this->train_data_->NumTotalBin() * sizeof(HistogramBinEntry);

  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  is_feature_aggregated_.resize(this->num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  buffer_write_start_pos_.resize(this->num_features_);
  buffer_read_start_pos_.resize(this->num_features_);
  global_data_count_in_leaf_.resize(this->tree_config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::ResetConfig(const TreeConfig* tree_config) {
  TREELEARNER_T::ResetConfig(tree_config);
  global_data_count_in_leaf_.resize(this->tree_config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // generate feature partition for current tree	为当前的树生成特征划分
  // feature_distribution[机器索引] = Vector<int>	(元素为内部特征索引)
  // num_bins_distributed[机器索引] = 当前机器中分配到的所有特征的所有bin的总数
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);										//每个机器上的bin数目
  for (int i = 0; i < this->train_data_->num_total_features(); ++i) {							//遍历所有特征
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1) { continue; }
    if (this->is_feature_used_[inner_feature_index]) {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));		//获取bin数量最少的结点
      feature_distribution[cur_min_machine].push_back(inner_feature_index);						//将特征索引存储到对应节点
      auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
      if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetDefaultBin() == 0) {
        num_bin -= 1;
      }
      num_bins_distributed[cur_min_machine] += num_bin;											//更新节点上的总bin数
    }
    is_feature_aggregated_[inner_feature_index] = false;
  }
  // get local used feature										获取本地使用的特征
  for (auto fid : feature_distribution[rank_]) {
    is_feature_aggregated_[fid] = true;
  }

  // get block start and block len for reduce scatter			获取用于reduce scatter的块起始索引和块长度
  reduce_scatter_size_ = 0;
  for (int i = 0; i < num_machines_; ++i) {
    block_len_[i] = 0;
    for (auto fid : feature_distribution[i]) {					//获取每个节点中的所有特征索引
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0) {
        num_bin -= 1;
      }
      block_len_[i] += num_bin * sizeof(HistogramBinEntry);		//计算每个节点的块大小（即当前节点存储的bin的总大小）
    }
    reduce_scatter_size_ += block_len_[i];
  }

  block_start_[0] = 0;											//计算块的起始位置(当前块的起始位置=上一个块的起始位置+上一个块的块大小)
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
  }

  // get buffer_write_start_pos_	(所有节点)
  int bin_size = 0;
  for (int i = 0; i < num_machines_; ++i) {						//遍历所有节点
    for (auto fid : feature_distribution[i]) {					//遍历每一个节点中的所有特征索引
      buffer_write_start_pos_[fid] = bin_size;
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0) {
        num_bin -= 1;
      }
      bin_size += num_bin * sizeof(HistogramBinEntry);			//保存特征起始位置一直累加计算，即保存在一个连续空间里面
    }
  }

  // get buffer_read_start_pos_		(本地节点)
  bin_size = 0;
  for (auto fid : feature_distribution[rank_]) {		//遍历当前节点的特征索引
    buffer_read_start_pos_[fid] = bin_size;
    auto num_bin = this->train_data_->FeatureNumBin(fid);
    if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0) {
      num_bin -= 1;
    }
    bin_size += num_bin * sizeof(HistogramBinEntry);			//保存本地保存的所有特征的每个特征的访问起始位置
  }

  // sync global data sumup info			同步全局数据的信息总和
  std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(),
                                               this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  int size = sizeof(data);
  std::memcpy(input_buffer_.data(), &data, size);
  // global sumup reduce					全局加和reduce
  Network::Allreduce(input_buffer_.data(), size, size, output_buffer_.data(), [](const char *src, char *dst, int len) {
    int used_size = 0;
    int type_size = sizeof(std::tuple<data_size_t, double, double>);
    const std::tuple<data_size_t, double, double> *p1;
    std::tuple<data_size_t, double, double> *p2;
    while (used_size < len) {
      p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
      p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
	  // 获取元组中的对应元素并相加
      std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
      std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
      std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
      src += type_size;
      dst += type_size;
      used_size += type_size;				//累计一条元组的大小
    }
  });
  // copy back
  std::memcpy(&data, output_buffer_.data(), size);
  // set global sumup info
  this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data));
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplits() {
  TREELEARNER_T::ConstructHistograms(this->is_feature_used_, true);
  // construct local histograms				构造本地直方图
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if ((!this->is_feature_used_.empty() && this->is_feature_used_[feature_index] == false)) continue;
    // copy to buffer						复制到缓冲区
    std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
                this->smaller_leaf_histogram_array_[feature_index].RawData(),
                this->smaller_leaf_histogram_array_[feature_index].SizeOfHistgram());
  }
  // Reduce scatter for histogram			对直方图进行reduce scatter操作
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, block_start_.data(),
                         block_len_.data(), output_buffer_.data(), &HistogramBinEntry::SumReducer);
  this->FindBestSplitsFromHistograms(this->is_feature_used_, true);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t>&, bool) {
  std::vector<SplitInfo> smaller_bests_per_thread(this->num_threads_, SplitInfo());
  std::vector<SplitInfo> larger_bests_per_thread(this->num_threads_, SplitInfo());

  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_aggregated_[feature_index]) continue;
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    // restore global histograms from buffer
    this->smaller_leaf_histogram_array_[feature_index].FromMemory(
      output_buffer_.data() + buffer_read_start_pos_[feature_index]);

    this->train_data_->FixHistogram(feature_index,
                                    this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
                                    GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->LeafIndex()),
                                    this->smaller_leaf_histogram_array_[feature_index].RawData());
    SplitInfo smaller_split;
    // find best threshold for smaller child
    this->smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      this->smaller_leaf_splits_->sum_gradients(),
      this->smaller_leaf_splits_->sum_hessians(),
      GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->LeafIndex()),
      &smaller_split);
    smaller_split.feature = real_feature_index;
    if (smaller_split > smaller_bests_per_thread[tid]) {
      smaller_bests_per_thread[tid] = smaller_split;
    }

    // only root leaf		
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->LeafIndex() < 0) continue;

    // construct histgroms for large leaf, we init larger leaf as the parent, so we can just subtract the smaller leaf's histograms
    this->larger_leaf_histogram_array_[feature_index].Subtract(
      this->smaller_leaf_histogram_array_[feature_index]);
    SplitInfo larger_split;
    // find best threshold for larger child
    this->larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      this->larger_leaf_splits_->sum_gradients(),
      this->larger_leaf_splits_->sum_hessians(),
      GetGlobalDataCountInLeaf(this->larger_leaf_splits_->LeafIndex()),
      &larger_split);
    larger_split.feature = real_feature_index;
    if (larger_split > larger_bests_per_thread[tid]) {
      larger_bests_per_thread[tid] = larger_split;
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->LeafIndex();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr &&  this->larger_leaf_splits_->LeafIndex() >= 0) {
    leaf = this->larger_leaf_splits_->LeafIndex();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_bests_per_thread);
    this->best_split_per_leaf_[leaf] = larger_bests_per_thread[larger_best_idx];
  }

  SplitInfo smaller_best_split, larger_best_split;
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()];
  // find local best split for larger leaf		为更大的叶子查找出本地最优的划分
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()];
  }

  // sync global best info						同步全局最优划分信息
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->tree_config_->max_cat_threshold);

  // set best split								设置最优划分
  this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()] = smaller_best_split;
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()] = larger_best_split;
  }
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  TREELEARNER_T::Split(tree, best_Leaf, left_leaf, right_leaf);
  const SplitInfo& best_split_info = this->best_split_per_leaf_[best_Leaf];
  // need update global number of data in leaf				需要更新叶子上的全局数据的数量
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
}

// instantiate template classes, otherwise linker cannot find the code		初始化模板类
template class DataParallelTreeLearner<GPUTreeLearner>;
template class DataParallelTreeLearner<SerialTreeLearner>;

}  // namespace LightGBM
