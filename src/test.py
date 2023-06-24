import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from dataGenerator import DataGenerator

dataPath = "/path/to/test/data"
test_csv_path = dataPath + "/test/test.csv"

test_df = pd.read_csv(test_csv_path)

# Set the image directory and extension
img_dir = dataPath + "/test"
img_ext = "bmp"

# Create a DataGenerator for the test data
test_generator = DataGenerator(
    df=test_df,
    img_dir=img_dir,
    img_ext=img_ext,
    batch_size=32,
    shuffle=False
)

model_path = "/path/to/saved/model"
model = load_model(model_path)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
