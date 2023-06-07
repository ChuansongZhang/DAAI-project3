from .bisenetv2 import BiSeNetV2
from .mobilenetv2 import MobileNetV2
from .deeplabv3 import deeplabv3_mobilenetv2


def make_model(args):
    if args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes, output_aux=args.output_aux, pretrained=args.pretrained)
    elif args.model == 'deeplabv3':

        dict_model = {
            'deeplabv3': {'model': deeplabv3_mobilenetv2, 'kwargs': {}},
        }
        # if args.hp_filtered and augm_model:
        #   dict_model[args.model]['kwargs']['in_channels'] = 4

        model = dict_model[args.model]['model'](args.num_classes, **dict_model[args.model]['kwargs'])

    else:
        raise NotImplementedError("Specify a correct model.")

    return model
