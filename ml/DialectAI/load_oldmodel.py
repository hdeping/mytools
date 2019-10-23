import torch
from mymodel import LanNet
# 优化器，SGD更新梯度
def getModel(dimension,language_nums):
    train_module = LanNet(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=language_nums)
    model = LanNet(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=9)
    #logging.info(model)
    model.load_state_dict(torch.load('./models/model18.model'))
    #old_keys = model.state_dict().keys()
    old_keys = model.state_dict().keys()
    keys = []
    # convert to the subscripted array
    for key in old_keys:
        keys.append(key)
    #print(keys)
    # new dict
    dict_new = train_module.state_dict().copy()
    for key in keys[:-2]:
        #print(key)
        dict_new[key] =  model.state_dict()[key]
    # load parameter
    train_module.load_state_dict(dict_new)
    return train_module
#train_module = getModel(40,10)
