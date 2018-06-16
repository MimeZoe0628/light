#ifndef LIGHTGBM_TREELEARNER_LEAF_SPLITS_HPP_
#define LIGHTGBM_TREELEARNER_LEAF_SPLITS_HPP_

#include <LightGBM/meta.h>
#include "data_partition.hpp"

#include <vector>

namespace LightGBM {

/*!
* \brief used to find split candidates for a leaf	用来查找一个叶子的划分候选
*/
class LeafSplits {
public:
  LeafSplits(data_size_t num_data)
    :num_data_in_leaf_(num_data), num_data_(num_data),
    data_indices_(nullptr) {
  }
  void ResetNumData(data_size_t num_data) {
    num_data_ = num_data;
    num_data_in_leaf_ = num_data;
  }
  ~LeafSplits() {
  }

  /*!
  * \brief Init split on current leaf on partial data. 
  * \param leaf Index of current leaf
  * \param data_partition current data partition
  * \param sum_gradients
  * \param sum_hessians
  */
  void Init(int leaf, const DataPartition* data_partition, double sum_gradients, double sum_hessians) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
  }

  /*!
  * \brief Init splits on current leaf, it will traverse all data to sum up the results
  * \param gradients
  * \param hessians
  */
  void Init(const score_t* gradients, const score_t* hessians) {
    num_data_in_leaf_ = num_data_;
    leaf_index_ = 0;
    data_indices_ = nullptr;
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
#pragma omp parallel for schedule(static) reduction(+:tmp_sum_gradients, tmp_sum_hessians)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      tmp_sum_gradients += gradients[i];
      tmp_sum_hessians += hessians[i];
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
  }

  /*!
  * \brief Init splits on current leaf of partial data.
  * \param leaf Index of current leaf
  * \param data_partition current data partition
  * \param gradients
  * \param hessians
  */
  void Init(int leaf, const DataPartition* data_partition, const score_t* gradients, const score_t* hessians) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
#pragma omp parallel for schedule(static) reduction(+:tmp_sum_gradients, tmp_sum_hessians)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      data_size_t idx = data_indices_[i];
      tmp_sum_gradients += gradients[idx];
      tmp_sum_hessians += hessians[idx];
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
  }


  /*!
  * \brief Init splits on current leaf, only update sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  */
  void Init(double sum_gradients, double sum_hessians) {
    leaf_index_ = 0;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
  }

  /*!
  * \brief Init splits on current leaf
  */
  void Init() {
    leaf_index_ = -1;
    data_indices_ = nullptr;
    num_data_in_leaf_ = 0;
  }


  /*! \brief Get current leaf index */
  int LeafIndex() const { return leaf_index_; }

  /*! \brief Get numer of data in current leaf */
  data_size_t num_data_in_leaf() const { return num_data_in_leaf_; }

  /*! \brief Get sum of gradients of current leaf */
  double sum_gradients() const { return sum_gradients_; }
  
  /*! \brief Get sum of hessians of current leaf */
  double sum_hessians() const { return sum_hessians_; }

  /*! \brief Get indices of data of current leaf */
  const data_size_t* data_indices() const { return data_indices_; }


private:
  /*! \brief current leaf index 当前叶子索引*/
  int leaf_index_;
  /*! \brief number of data on current leaf 当前叶子包含的数据量 */
  data_size_t num_data_in_leaf_;
  /*! \brief number of all training data 所有训练数据的数量*/
  data_size_t num_data_;
  /*! \brief sum of gradients of current leaf 当前叶子上数据的一阶导数之和*/
  double sum_gradients_;
  /*! \brief sum of hessians of current leaf 当前叶子上数据的二阶导数之和 */
  double sum_hessians_;
  /*! \brief indices of data of current leaf 当前叶子上包含的数据索引*/
  const data_size_t* data_indices_;
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_LEAF_SPLITS_HPP_
