import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self, num_classes=11):
        super(VGG11, self).__init__()
        self.features = self.make_feature_extractors() # we define feature extractor, layers
        self.classifier = nn.Linear(512, num_classes) # 40 : the total number of labels to predict
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
    
    def make_feature_extractors(self):
        layers = []
        layers_configuration = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,  512, 'M']
        in_channels = 1 # We work with single channel images (grayscale)
        for layer in layers_configuration:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, layer, kernel_size=3, padding=1),
                           nn.BatchNorm2d(layer),
                           nn.ReLU(inplace=True) # Activation function
                           ]
                in_channels = layer
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)] # Average pooling
        return nn.Sequential(*layers)