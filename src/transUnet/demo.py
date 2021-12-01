import torch
from unet.unet_transformer.unet import TransUnet
from torchsummary import summary

a = torch.rand(2, 3, 256, 256)

model = TransUnet(in_channels=3, img_dim=256, vit_blocks=4, vit_dim_linear_mhsa_block=512, classes=1)
summary(model, input_size=(3, 256, 256), batch_size=8, device='cpu')
y = model(a)
print('final out shape:', y.shape)
