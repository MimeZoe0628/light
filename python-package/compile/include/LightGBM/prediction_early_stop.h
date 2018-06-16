#ifndef LIGHTGBM_PREDICTION_EARLY_STOP_H_
#define LIGHTGBM_PREDICTION_EARLY_STOP_H_

#include <functional>
#include <string>

#include <LightGBM/export.h>

namespace LightGBM {

struct PredictionEarlyStopInstance {
  /// Callback function type for early stopping.						用于提前终止的回调函数
  /// Takes current prediction and number of elements in prediction		取当前的预测以及预测过程中的元素数
  /// @returns true if prediction should stop according to criterion	如果判别条件指示应该停止则返回true
  using FunctionType = std::function<bool(const double*, int)>;

  FunctionType callback_function;  // callback function itself
  int          round_period;       // call callback_function every `runPeriod` iterations
};

struct PredictionEarlyStopConfig {
  int round_period;
  double margin_threshold;
};

/// Create an early stopping algorithm of type `type`, with given round_period and margin threshold
LIGHTGBM_EXPORT PredictionEarlyStopInstance CreatePredictionEarlyStopInstance(const std::string& type,
                                                                              const PredictionEarlyStopConfig& config);

}   // namespace LightGBM

#endif // LIGHTGBM_PREDICTION_EARLY_STOP_H_
