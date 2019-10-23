import torch

from net_component import LanNet
language_nums = 6
dimension = 40
train_module = LanNet(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=language_nums)

print(dir(train_module.parameters()))
#train_module.load_state_dict(torch.load("models_data10/model9.model"))
for param in train_module.parameters():
    #print(param.data)
    print(param.data.shape)
print("new models")
train_module.load_state_dict(torch.load("models_data10/model9.model"))
for param in train_module.parameters():
    print(param.data)
    #print(param.data.shape)
