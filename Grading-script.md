## Steps

1. Setup python environment using requirments.txt file
2. Download the pretraiend model from the google [drive link](https://drive.google.com/drive/folders/1JltX6UmexEq7gA02n0jPlXCcOSl5sxNi?usp=sharing)

3. Organise the pretrained weights directory as follows:
```Shell
  ./weights/Pretrained-models/
    Pretrained_baseModel_MobileNet.pth
    Pretrained_baseModel_ResNet50.pth

    Pretrained_ourModel_MobileNet.pth
    Pretrained_ourModel_ResNet50.pth
    Pretrained_ourModel_ResNet50_mse.pth
```

We also stored their corresponding PR curves data in the same google drive folder [ending with .pkl extension]

4. You also need to place the **dataset** as directed in the README.md file





## Evaluation widerface val
### 1. Generate txt files, calculate and saved PR curves data [.pickle format] in the pretrained model directory.

```Shell
python test_widerface.py --trained_model './weights/Pretrained-models/Pretrained_ourModel_ResNet50.pth' --save_folder './widerface_evaluate/txt_files/'
```

Where, 

- **trained_model**: Path to the pretrained model
- **save_folder**: Dir to save txt result files



### 2. Plot PR curves
```Shell
python plot_results.py --network "resnet50" --IoU_lossFunction "bce"
```

Where, 

- **network**: Backbone network "mobile0.25" or "resnet50"
- **IoU_lossFunction**: IoU loss computation function "bce" or "mse" [use only for "resnet50"]

The plots will be saved in the **outputs/** folder in the project directory

