import torch
import timm
import os

color_path = r"D:\Chula\Senior_Project\Phase_d03_best.pth"
defect_path = r"D:\Chula\Senior_Project\Phase3_WD0.15_best.pth"

device = torch.device('cpu')
BASE_CONVNEXT_MODEL = "convnext_tiny"

print("Loading PyTorch models for export...")
model_color = timm.create_model(BASE_CONVNEXT_MODEL, pretrained=False, num_classes=2)
model_color.load_state_dict(torch.load(color_path, map_location=device))
model_color.eval()

model_defect = timm.create_model(BASE_CONVNEXT_MODEL, pretrained=False, num_classes=4)
model_defect.load_state_dict(torch.load(defect_path, map_location=device))
model_defect.eval()

dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting Color model to ONNX...")
torch.onnx.export(
    model_color, dummy_input,
    r"D:\Chula\Senior_Project\Phase_d03_best.onnx",
    export_params=True, opset_version=14,
    do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Exporting Defect model to ONNX...")
torch.onnx.export(
    model_defect, dummy_input,
    r"D:\Chula\Senior_Project\Phase3_WD0.15_best.onnx",
    export_params=True, opset_version=14,
    do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✅ ONNX export complete.")
