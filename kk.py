import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2()
model.load_state_dict(torch.load('/Users/marc-antoine/Downloads/mobilenet_v2-b0353104.pth'))
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
torchscript_model_optimized = optimize_for_mobile(traced_script_module)
torchscript_model_optimized._save_for_lite_interpreter("model.pt")