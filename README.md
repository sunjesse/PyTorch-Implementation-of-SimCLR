# SimCLR_pytorch
PyTorch implementation of arxiv.org/pdf/2002.05709.pdf. 

This is a simple framework for contrastive learning of visual representations that comprises of the following modules:
  1. Sequential data augmentations (Crop, flip, ..., color distortion, Gaussian blur)
  2. A base encoder network (ResNet18 is used by default in this implementation)
  3. A projection head network (just a fully connected layer at the end of the base encoder)
  4. A contrastive loss implemented as SimLoss
