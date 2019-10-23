import torch
from mymodel import pre_model

model = pre_model()
model_name = "models0/model39-0.model"
model.load_state_dict(torch.load(model_name))
