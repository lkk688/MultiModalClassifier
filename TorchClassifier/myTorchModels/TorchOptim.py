import torch
import torch.nn as nn
import torch.optim as optim

#different optim: ref: https://ruder.io/optimizing-gradient-descent/
def gettorchoptim(name, model_ft, lr=0.001, momentum=0.9):
    if name=='SGD':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    elif name=='Adam':
        optimizer_ft = optim.Adam(model_ft.parameters())
    return optimizer_ft