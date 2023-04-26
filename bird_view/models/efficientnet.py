"""from efficientnet_pytorch.model import EfficientNet
import torch

model_funcs = {
        'efficientnet-b0': 1280,
        'efficientnet-b7': 2560,
        }

def get_efficientnet(model_name='efficientnet-b1', pretrained=False):
    c_out = model_funcs[model_name]
    if pretrained:
        model = EfficientNet.from_pretrained(model_name)
        return model, c_out
    model = EfficientNet.from_name(model_name)
    return model, c_out"""


from efficientnet_pytorch import EfficientNet
import torch
model_name = 'efficientnet-b6'
model = EfficientNet.from_pretrained(model_name)
model_state_dict = model.state_dict()
model_state_dict.pop('_fc.weight')
model_state_dict.pop('_fc.bias')
torch.save(model_state_dict, '/home/carla/Desktop/LearningByCheating/bird_view/models/efficientnet_pretrained/{}.th'.format(model_name))

