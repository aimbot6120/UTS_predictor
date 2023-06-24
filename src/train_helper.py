import numpy as np

def compute_metrics(model, train_generator,val_generator):
    """Compute evaluation metrics for the model"""
    train_loss = model.evaluate(train_generator)
    val_loss = model.evaluate(val_generator)
    return train_loss, val_loss

def plot_history(history):
    """Plot the training history of the model"""
    import matplotlib.pyplot as plt
    
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
