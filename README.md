# cDDPM

## Dataset

The model is set up to use a modified version of the CelebA dataset, downscaled to 64x64 pixels. To download the dataset, follow these steps:

1. Open a terminal.
2. Run the following command to download the dataset: $kaggle datasets download -d ahmedshawaf/celeba
3. Once the download is complete, unzip the dataset: $unzip celeba.zip
4. Run the data_processing.py script to preprocess the dataset: $python3 data_processing.py


## Training the Model

To train the model, use the train.py script. Before running the training script, ensure that the dataset has been downloaded and preprocessed as mentioned in the previous section.

The main details and configurations of the model can be found in the cUNet.py file.

## Additional Experiments

The alt_diffusion.ipynb notebook contains an extended version of the project, including various small experiments and the ability to change datasets more easily. This notebook provides a convenient way to iterate and explore different experiments.
