from models.unet_model import UNet


def get_model(model, input_nc, output_nc):
    if model == 'unet':
        return UNet(input_nc, output_nc)
    else:
        raise Exception('Model cannot be found')
