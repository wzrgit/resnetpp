#include "resnet.h"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <codecvt>
#include <fstream>

namespace resnetpp {

struct Model {
  Ort::Session session{nullptr};
};

ResNet::ResNet(const ResNet::Config& config) {
  Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ResNet");
  Ort::SessionOptions session_options;
  session_options.SetInterOpNumThreads(config.threads_number);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _MSC_VER
  std::wstring w_model_path(config.model_path.begin(), config.model_path.end());
  const wchar_t* m_path = w_model_path.c_str();
#else
  const char* m_path = config.model_path.c_str();
#endif

  Ort::Session session(env, m_path, session_options);

  model_ = std::make_unique<Model>();
  model_->session = std::move(session);

  if (!config.labels_path.empty()) {
    LoadLabels(config.labels_path);
  }

  std::vector<int64_t> shape = GetInputShape();
  channels_ = SHAPE_CHANNEL(shape);
  height_ = SHAPE_H(shape);
  width_ = SHAPE_W(shape);
}

ResNet::~ResNet() {}

std::vector<int64_t> ResNet::GetInputShape() const {
  return model_->session.GetInputTypeInfo(0)
      .GetTensorTypeAndShapeInfo()
      .GetShape();
}

std::vector<ResNet::Result> ResNet::Inference(
    const std::vector<uint8_t>& inputs,
    int top_size) {
  std::vector<float> input_float = PreProcessImage(inputs);

  std::vector<int64_t> input_shape = {1, channels_, height_, width_};
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, input_float.data(), input_float.size(), input_shape.data(),
      input_shape.size());

  const char* input_names[] = {"input"};
  const char* output_names[] = {"output"};

  std::vector<Ort::Value> outputs_tensors = model_->session.Run(
      Ort::RunOptions(), input_names, &input_tensor, 1, output_names, 1);

  float* output_data = outputs_tensors[0].GetTensorMutableData<float>();
  auto output_shape = outputs_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t output_size = 1;
  for (auto& s : output_shape) {
    output_size *= s;
  }

  std::vector<float> outputs(output_data, output_data + output_size);

  std::vector<size_t> indices(outputs.size());
  for (size_t i = 0; i < indices.size(); i++) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), [&outputs](size_t i1, size_t i2) {
    return outputs[i1] > outputs[i2];
  });

  float max_val = *std::max_element(outputs.begin(), outputs.end());
  float min_val = *std::min_element(outputs.begin(), outputs.end());
  for (auto& e : outputs) {
    e = (e - min_val) / (max_val - min_val);
  }

  size_t x = top_size <= 0 ? outputs.size() : top_size;

  x = std::min(x, outputs.size());
 
  std::vector<ResNet::Result> result(x);
  for (size_t i = 0; i < x; i++) {
    Result& r = result[i];
    r.class_id = (int)indices[i];
    r.prob = outputs[r.class_id];
    if (auto it = labels_.find(r.class_id); it != labels_.end()) {
      r.label = it->second;
    }
  }

  return result;
}

std::vector<float> ResNet::PreProcessImage(const std::vector<uint8_t>& inputs) {
  int64_t sizes = height_ * width_ * channels_;

  if (sizes != inputs.size()) {
    // TODO error
    return std::vector<float>();
  }

  std::vector<float> result(sizes);
  constexpr float mean[] = {0.485f, 0.456f, 0.406f};
  constexpr float std[] = {0.229f, 0.224f, 0.225f};

  int64_t step = height_ * width_;

  for (int64_t i = 0; i < step; i++) {
    for (int64_t c = 0; c < 3; c++) {
      int64_t idx = i + c * step;
      result[idx] = inputs[idx] / 255.0f;
      result[idx] = (result[idx] - mean[c]) / std[c];
    }
  }

  return result;
}

void ResNet::LoadLabels(const std::string& labels_path) {
  labels_.clear();
  std::ifstream ifs;
  ifs.open(labels_path, std::ios_base::in);
  if (!ifs.is_open()) {
    // TODO log
    return;
  }

  std::string line;
  int idx = 0;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      break;
    }
    labels_.insert({idx, line});
    ++idx;
  }
  ifs.close();
}

}  // namespace resnetpp