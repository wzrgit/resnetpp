#ifndef _RESNET_CPP_RESNET_H_
#define _RESNET_CPP_RESNET_H_

#include "resnet_export.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#define SHAPE_H(shape) shape[2]
#define SHAPE_W(shape) shape[3]
#define SHAPE_CHANNEL(shape) shape[1]

namespace resnetpp {

struct Model;

class RESNET_API ResNet {
 public:
  enum DEIVCE_TYPE {
    CPU = 0,
    CUDA = 1,
  };

  struct Config {
    std::string model_path;
    std::string labels_path;
    DEIVCE_TYPE device_type = CPU;
    int threads_number = 1;
  };

  struct Result {
    int class_id;
    std::string label;
    float prob;  // 0~1
  };

  ResNet(const Config& config);
  ResNet(const ResNet&) = delete;
  ~ResNet();

  std::vector<int64_t> GetInputShape() const;

  // inputs size: h * w * channels  e.g 224 * 224 * 3
  // [R,R,R...,G,G,G...,B,B,B]
  std::vector<Result> Inference(const std::vector<uint8_t>& inputs,
                                int top_size);

 private:
  std::vector<float> PreProcessImage(const std::vector<uint8_t>& inputs);

  void LoadLabels(const std::string& labels_path);

 private:
  std::unique_ptr<Model> model_;
  std::map<int, std::string> labels_;

  int64_t channels_ = 0;
  int64_t height_ = 0;
  int64_t width_ = 0;
};

}  // namespace resnetpp

#endif  // _RESNET_CPP_RESNET_H_