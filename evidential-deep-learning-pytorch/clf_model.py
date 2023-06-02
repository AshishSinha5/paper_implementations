import torch
import torch.nn as nn


class clf_model(nn.Module):
    def __init__(self, units:list, n_classes:int = 10):
        super().__init__()
        self.units = units
        self.n_layers = len(self.units)
        self.n_classes = n_classes
        self.output = nn.Softmax(dim=1)
        
        # layers
        self.layers = nn.ModuleList()
        self.get_layers()
        ## loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        print(f'Model Intialized: {self}')


    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        x = self.output(x)

        return x

    def loss(self, pred, label):
        loss = self.cross_entropy_loss(pred, label)
        return loss
    
    def get_layers(self):

        for i in range(len(self.units)-1):
            layer = nn.Linear(self.units[i], self.units[i+1])
            self.layers.append(layer)
        
        layer = nn.Linear(self.units[-1], self.n_classes)
        self.layers.append(layer)