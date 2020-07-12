# CV_Image_Captioning
CNN-LSTM Image Captioning Project

The model and hyperparameters in this repository are informed by  [this paper](https://arxiv.org/pdf/1502.03044.pdf) and [this paper](https://arxiv.org/pdf/1411.4555.pdf).

My image-captioning architecture consists of two pieces. The first is the encoder. The encoder's first section is a pretrained ResNet50 CNN. The next layer converts the features output from that CNN into a 512-length embedding using a Linear layer. This 512-length vector is then fed into the next block. The second block of the overall architecture, which the output of the encoder goes to, is the decoder. The decoder's first layer is an LSTM layer that takes in the embedding from the decoder and creates an output and a hidden state that have a size of 512. The output of the LSTM layer goes to a Linear layer to produce an output of scores for each token in the vocabulary. I used https://arxiv.org/pdf/1411.4555.pdf as the basis for my architecture andother hyperparameters, except for the optimizer. I used https://arxiv.org/pdf/1502.03044.pdf as the basis for optimizer choice.

I chose all the weights to be trained except those that came from the base CNN network. The reason for this is because the base CNN had already been trained to create a set of
features that was sufficiently good at discriminating images. Letting those weights train would be an inefficient time cost.

I chose the Adam optimizer based on the "Show, Attend and Tell: Neural Image Caption
Generation with Visual Attention" paper referenced above that also found that for the COCO dataset the Adam optimizer gave the best performance. 

An image transform `transform_train` for pre-processing the training images is also provided, but you are welcome to modify it as you wish.  When modifying this transform, keep in mind that:
- the images in the dataset have varying heights and widths, and 
- if using a pre-trained model, you must perform the corresponding appropriate normalization.

I have transform_train resize the data, take random crops that are the size that Encoder expects, randomly flips data horizontally, converts the data into a tensor for PyTorch to use, normalizes the images based on the mean and standard deviation for the RGB channels.