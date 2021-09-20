import torch
import torch.nn as nn
import torch.optim as optim

#different optim: ref: https://ruder.io/optimizing-gradient-descent/
def gettorchoptim(name, model_ft, lr=0.001, momentum=0.9):
    if name=='SGD':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    elif name=='Adam':
        optimizer_ft = optim.Adam(model_ft.parameters())
    elif name=='adamresnetcustomrate':
        optimizer_ft = adamresnetcustomrate(model_ft)
    return optimizer_ft

#discriminative fine-tuning - a technique used in transfer learning where later layers in a model have higher learning rates than earlier ones.
def adamresnetcustomrate(model, FOUND_LR = 1e-3):
    params = [
          {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
          {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
          {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
          {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
          {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
          {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
          {'params': model.fc.parameters()}
         ]


    optimizer = optim.Adam(params, lr = FOUND_LR)
    return optimizer