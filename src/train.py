import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from data_generator import DataGenerator
from model import build_model
from train_helper import compute_metrics, plot_history

tf.random.set_seed(42)

parser = argparse.ArgumentParser(description="Train the steel strength prediction model")
parser.add_argument("--csv-path", type=str, default="data/sample.csv",
                    help="path to the CSV file containing the sample data")
parser.add_argument("--image-dir", type=str, default="data/images",
                    help="path to the directory containing the sample images")
parser.add_argument("--epochs", type=int, default=3,
                    help="number of epochs to train the model for")
parser.add_argument("--batch-size", type=int, default=32,
                    help="batch size to use during training")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate for the optimizer")
parser.add_argument("--save-dir", type=str, default="models",
                    help="directory to save the trained model")
args = parser.parse_args()

# Create the save directory if it doesn't exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

df = pd.read_csv(args.csv_path)

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# print(train_df)
# print(train_df.loc[0,"alloy_id"])
# print(train_df.loc[0,"sample_id"])
# Create the data generator
train_generator = DataGenerator(df = train_df,img_dir=args.image_dir,
                                batch_size=args.batch_size,img_ext="bmp")
val_generator = DataGenerator(df = val_df, img_dir=args.image_dir,
                                batch_size=args.batch_size,img_ext="bmp")
model = build_model()
model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(lr=args.lr))

history = model.fit(
        x = train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback, csv_logger],
        verbose=1,
        shuffle=False,
    )

train_loss, val_loss = compute_metrics(model, train_generator,val_generator)

# Save the trained model
model.save(os.path.join(args.save_dir, "steel_strength_model.h5"))

# Print final evaluation metrics
print("Train Loss:", train_loss)
print("Validation Loss:", val_loss)

# Plot training history
plot_history(history)