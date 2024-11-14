#include "resnet.h"

#include <opencv2/opencv.hpp>

int main() {
  const std::string model_path = "model/resnet-50.onnx";
  const std::string labels_path = "model/labels.txt";
  resnetpp::ResNet::Config cfg;
  cfg.model_path = model_path;
  cfg.labels_path = labels_path;
  resnetpp::ResNet resnet(cfg);

  const char* img_path = "panda.jpg";
  cv::Mat img = cv::imread(img_path);
  assert(!img.empty());

  // BGR
  int channels = img.channels();
  assert(channels >= 3);

  std::vector<int64_t> shape = resnet.GetInputShape();

  int64_t h = SHAPE_H(shape);
  int64_t w = SHAPE_W(shape);
  int64_t c = SHAPE_CHANNEL(shape);

  cv::Mat img_resized;
  cv::resize(img, img_resized, cv::Size(w, h));

  std::vector<uint8_t> inputs(w * h * c, 0);

  int steps = h * w;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int idx_r = i * w + j;
      int idx_g = idx_r + steps;
      int idx_b = idx_r + 2 * steps;

      cv::Vec3b p = img_resized.at<cv::Vec3b>(i, j);
      inputs[idx_b] = p[0];
      inputs[idx_g] = p[1];
      inputs[idx_r] = p[2];
    }
  }

  auto result = resnet.Inference(inputs, 5);
  for (const auto& r: result){
    printf("%3d%% be [%d] %s\n", (int)(r.prob * 100),r.class_id, r.label.c_str());
  }

  return 0;
}