from mymodel import LanNet
import torch

dimension = 40
language_nums = 10

train_module = LanNet(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=language_nums)
train_module.load_state_dict(torch.load("models/model9.model"))
print(train_module)
for key in train_module.state_dict():
    print(key)

