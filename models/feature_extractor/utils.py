import torch
def build_models(model_type):
    if model_type == 'vgg11':
        from models.feature_extractor.vgg import VGG
        net = VGG('VGG11')
    elif model_type == 'vgg13':
        from models.feature_extractor.vgg import VGG
        net = VGG('VGG13')
    elif model_type == 'vgg16':
        from models.feature_extractor.vgg import VGG
        net = VGG('VGG16')
    elif model_type == 'vgg19':
        from models.feature_extractor.vgg import VGG
        net = VGG('VGG19')
    elif model_type == 'resnet18':
        from models.feature_extractor.resnet import ResNet18
        net = ResNet18()
    elif model_type == 'resnet34':
        from models.feature_extractor.resnet import ResNet34
        net = ResNet34()
    elif model_type == 'resnet50':
        from models.feature_extractor.resnet import ResNet50
        net = ResNet50()
    elif model_type == 'resnet101':
        from models.feature_extractor.resnet import ResNet101
        net = ResNet101()
    elif model_type == 'resnet152':
        from models.feature_extractor.resnet import ResNet152
        net = ResNet152()

    feature_dim = net.get_feature_dim()
    return net, feature_dim