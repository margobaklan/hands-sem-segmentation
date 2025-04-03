## Hands Semantic Segmentation

This repository provides a framework for training and testing hand segmentation models using the **EgoHands dataset** with **DeepLabV3** and **UNet** architectures. The project includes tools for preparing data, training models, and testing them in real-time using live camera feeds.

### Dataset
Download the Labeled Data EgoHands dataset from the official website down below and place it in the core folder.

https://vision.soic.indiana.edu/projects/egohands/

### Data Preparation

The command below will split the dataset with 80% for training and 20% for testing.
```
python prepare_data.py 0.8
```
### Training
To train DeepLabV3 or UNet architectures open and run the notebook `train_models.ipynb`.
The trained models will be saved to the dir `models/`

### Live Testing
Example testing the trained models with a live camera feed using:
```
python live_segm.py --model deeplab --checkpoint models/deeplab.pth
```
This also will save the video captured results.
