import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    @staticmethod
    def random_flip_and_rotate(img):
        if random.random() < 0.5:
            img = np.flipud(img)

        if random.random() < 0.5:
            img = np.fliplr(img)

        angle = random.choice([0, 1, 2, 3])
        img = np.rot90(img, angle)

        return img.copy()

    def preprocess_image(self, img, target_size):
        # Resize the image
        image = img.resize(target_size)
        # Convert image to numpy array
        image_array = np.array(image)
        # Normalize pixel values
        image_array = image_array / 255.0
        # If the image is grayscale, convert it to RGB
        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
        # Randomly flip and rotate image
        image_array = self.random_flip_and_rotate(image_array)
        return image_array.transpose(2, 0, 1)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        target_size = (240, 240)
        img_path = self.img_dir / Path(self.img_labels.loc[idx, 'full_path'].strip("'[]'"))
        image = Image.open(img_path)
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        new_image = Image.new('RGB', target_size)
        new_image.paste(image, ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2))
        label = self.img_labels.loc[idx, 'age']
        if self.transform:
            image = self.transform(np.array(new_image))
        image = self.preprocess_image(image, target_size)
        return image, label


class TestDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_ids = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def preprocess_image(self, img, target_size):
        # Resize the image
        image = img.resize(target_size)
        # Convert image to numpy array
        image_array = np.array(image)
        # Normalize pixel values
        image_array = image_array / 255.0
        # If the image is grayscale, convert it to RGB
        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
       
        return image_array.transpose(2, 0, 1)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        target_size = (240, 240)
        img_path = self.img_dir / Path(self.img_ids.loc[idx, 'full_path'].strip("'[]'"))
        image = Image.open(img_path)
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        new_image = Image.new('RGB', target_size)
        new_image.paste(image, ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2))
        if self.transform:
            image = self.transform(np.array(new_image))
        image = self.preprocess_image(image, target_size)
        return image, int((img_path.stem).strip("(), '"))

