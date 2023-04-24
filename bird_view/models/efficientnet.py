from efficientnet_pytorch.model import EfficientNet

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
    return model, c_out
