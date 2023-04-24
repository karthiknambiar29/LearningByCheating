from efficientnet_pytorch import EfficientNet

model_funcs = {
        'efficient-b0': 1280,
        'efficient-b7': 2560,
        }

def get_efficientnet(model_name='efficient-b1', pretrained=False):
    c_out = model_funcs[model_name]
    if pretrained:
        model = EfficientNet.from_pretrained(model_name)
        return model, c_out
    model = EfficientNet.from_name(model_name)
    return model, c_out