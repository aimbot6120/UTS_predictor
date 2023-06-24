import pandas as pd
import numpy as np
import cv2
import tensorflow.keras.utils as utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator(utils.Sequence):
    def __init__(self, df, img_dir, img_ext, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.shuffle = shuffle
        self.df = df

        # Drop unnecessary columns
        self.x = df.drop(['sample_id', 'alloy_id'], axis=1)

        num_samples = len(df)
        self.indices = np.arange(num_samples)
        np.random.shuffle(self.indices)

        # print(df["alloy_id"].values[0])
        # print(self.indices)


        # Split the data into features and target
        self.x = self.x.drop(['Ultimate Tensile Stress, [MPa]'] , axis=1)
        self.y = df['Ultimate Tensile Stress, [MPa]']
        
        # Normalize the features
        self.x = (self.x - self.x.mean()) / self.x.std()

        # Convert the data to numpy arrays
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        # Create an ImageDataGenerator for data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True
        )

    def __len__(self):
        if self.shuffle:
            return np.ceil(len(self.indices) / self.batch_size).astype(int)
        else:
            return np.ceil(len(self.x) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        indices = self.indices
        if self.shuffle:
            indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            indices = np.arange(len(self.x))[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Load and preprocess the image data
        x_img = []
        for i in self.indices:
            # print(i)
            img_path = f"{self.img_dir}/{self.df['alloy_id'].values[i]}/{self.df['sample_id'].values[0]}.{self.img_ext}"
            # print(img_path)
           
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = self.datagen.random_transform(img)
            x_img.append(img)
        x_img = np.array(x_img) / 255.0

        # Load and preprocess the chemical data
        x_chem = self.x[indices, 0:]
        y = self.y[indices]

        return [x_chem, x_img], y
