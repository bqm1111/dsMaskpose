import tensorrt as trt
from maskpose.maskpose import MaskPose
from maskpose.models import MergeResNet
from maskpose.maskpose import load_filtered_state_dict
from torchvision.models.resnet import BasicBlock
import os
import torch
import numpy as np
import torch.onnx

def preprocess(img, mean = 127.5, std = 128.0, device="cuda:0"):
    """preprocess image for inference"""
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32)
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    tensor = torch.from_numpy(img).to(device)
    return tensor

def softmax_temperature(tensor, temperature):
    """just softmax temperature"""
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result
# 
if __name__ == '__main__':
    model = MergeResNet(BasicBlock, [2, 2, 2, 2])
    snapshot = os.path.join(os.path.dirname(__file__),
                            "maskpose", "MaskPose_R18_310721.pkl")
    state_dict = torch.load(snapshot, map_location="cuda:0")
    load_filtered_state_dict(model, state_dict)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 112, 112)
    torch.onnx.export(model, dummy_input, "maskpose.onnx", input_names="face_input", output_names=["mask", "yaw", "pitch", "roll"], verbose=True, export_params=True)