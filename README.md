# face2age
Predicting age from images of faces

## Dependencies
See environment.yml

## Repo components

#### Files
- image_dataset.py
	- Costs the `ImageDataset` class to load and preprocess images and load labels
	- Called in `train.py` and `test.py`
- filter_data.py
	- Removes images with no labels 
- loss.py
	- Defines `Log Cosh` and `Hubosh` loss functions
	- Called in `train.py` and `test.py`
- network.py
	- Defines the `Model` class which is composed of the following
		- The `VGG16FeatureExtractor` class passes an input image through VGG16
		- The `AgePredictor` class is an MLP for regression of the VGG16 features to predict age
- train.py
	- Trains the `Model` class 
	- Splits training data into training, validation, and test sets
		- The test indices are saved to data/test_indices.txt
	- Saves the model weights, training metrics, and validation metrics to the `results` directory

#### Directories
- run
	- Contains `.sb` files to train and test the model on an HPC

