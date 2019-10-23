from mymodel import pre_model,LanNet
from testmodel import  inferModel as inferenceModel
import torch
import numpy as np

def old():
    model = pre_model()
    model_name = "models0/model39-0.model"
    model.load_state_dict(torch.load(model_name))

dimension = 40
language_nums = 10

def key2array(model_keys):
    keys = []
    i = 0
    for key in model_keys:
        #print(i,key)
        keys.append(key)
        i = i+1
    return keys

def getModel(dimension,language_nums):

    # define the models
    pre = pre_model()
    
    lannet = LanNet(input_dim=dimension, hidden_dim=512, bn_dim=64, output_dim=language_nums)
    
    infer = inferenceModel(input_dim=dimension, hidden_dim=512, bn_dim=64, output_dim=language_nums)
    
    # load state dict
    print("load resnet")
    model_name = "models/resnet.model"
    pre.load_state_dict(torch.load(model_name))
    
    print("load gru")
    model_name = "models/gru.model"
    lannet.load_state_dict(torch.load(model_name))
    
    # copy key
    
    # pre keys
    pre_keys = pre.state_dict().keys()
    pre_keys_array = key2array(pre_keys)
    # lannet keys
    lannet_keys = lannet.state_dict().keys()
    lannet_keys_array = key2array(lannet_keys)
    
    # copy the state dict
    dict_new = infer.state_dict().copy()
    
    #print("pre")
    #print(pre_keys_array)
    #print("infer")
    #print(infer.state_dict().keys())

    # copy pre keys: 0-84
    pre_keys_array = pre_keys_array[:85]
    print("loading pre parameter")
    for key in pre_keys_array:
        #print(key)
        dict_new[key] = pre.state_dict()[key]

    # copy lannet keys: all
    print("loading lannet parameter")
    for key in lannet_keys_array:
        #print(key)
        dict_new[key] = lannet.state_dict()[key]

    ## load parameters
    infer.load_state_dict(dict_new)

    return infer

infer = getModel(dimension,language_nums)
# save file
modelfile = "models/infer.model"
torch.save(infer.state_dict(), modelfile)
    
