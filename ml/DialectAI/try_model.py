from resnet import resnet_mymodel
import torch 

model = resnet_mymodel()
print(model)
b = torch.rand(100,1,400)

c = model(b)

#from modelCollection.resnet import resnet18
#
#model = resnet18()
#print(model)
#a = torch.rand(100,3,224,224)
#b = model(a)
#print(b.shape)
