import torch, os
from pathlib import Path
from transformers import ResNetForImageClassification

def convert_model(model_name, out_path):
  # processor = AutoImageProcessor.from_pretrained(model_name)
  model = ResNetForImageClassification.from_pretrained(model_name)
  model.eval()

  dummy_input = torch.randn(1, 3, 224, 224)
  dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

  model_out_path = os.path.join(out_path, Path(model_name).stem + ".onnx")
  labels_out_path = os.path.join(out_path, 'labels.txt')

  torch.onnx.export(
    model,
    dummy_input,
    model_out_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=dynamic_axes
  )

  with open(labels_out_path, 'w') as f:
    for _, v in model.config.id2label.items():
      f.write(v + '\n')


if __name__ == '__main__':
  model_name = 'microsoft/resnet-50'
  os.makedirs('model', exist_ok=True)
  convert_model(model_name, 'model')