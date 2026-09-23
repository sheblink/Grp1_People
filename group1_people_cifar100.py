"""Group 1: classify the CIFAR-100 'people' superclass.

Fine classes: baby, boy, girl, man, woman.
The script extracts the superclass, remaps the original CIFAR-100 fine labels
to 0-4, trains a CNN, evaluates it, and saves figures and the trained model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
PEOPLE_COARSE_ID = 14
ORIGINAL_FINE_IDS = np.array([2, 11, 35, 46, 98])
CLASS_NAMES = np.array(["baby", "boy", "girl", "man", "woman"])
EPOCHS = 30
BATCH_SIZE = 64

np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent / "group1_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_people_dataset():
    """Load CIFAR-100 and keep only examples whose coarse label is people."""
    (train_images, train_fine), (test_images, test_fine) = (
        keras.datasets.cifar100.load_data(label_mode="fine")
    )
    (_, train_coarse), (_, test_coarse) = keras.datasets.cifar100.load_data(
        label_mode="coarse"
    )

    # Convert labels from shape (N, 1) to shape (N,).
    train_fine = train_fine.squeeze()
    test_fine = test_fine.squeeze()
    train_coarse = train_coarse.squeeze()
    test_coarse = test_coarse.squeeze()

    train_mask = train_coarse == PEOPLE_COARSE_ID
    test_mask = test_coarse == PEOPLE_COARSE_ID

    x_train = train_images[train_mask]
    y_train_original = train_fine[train_mask]
    x_test = test_images[test_mask]
    y_test_original = test_fine[test_mask]

    # Map CIFAR-100 IDs [2, 11, 35, 46, 98] to model IDs [0, 1, 2, 3, 4].
    label_lookup = np.full(100, -1, dtype=np.int32)
    label_lookup[ORIGINAL_FINE_IDS] = np.arange(len(CLASS_NAMES))
    y_train = label_lookup[y_train_original]
    y_test = label_lookup[y_test_original]

    if np.any(y_train < 0) or np.any(y_test < 0):
        raise ValueError("Unexpected fine label found in the people superclass.")

    return x_train, y_train, x_test, y_test


def save_dataset_examples(images, labels):
    """Save one example from each of the five classes."""
    plt.figure(figsize=(10, 2.4))
    for class_id, class_name in enumerate(CLASS_NAMES):
        image_index = np.flatnonzero(labels == class_id)[0]
        plt.subplot(1, 5, class_id + 1)
        plt.imshow(images[image_index])
        plt.title(class_name)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_examples.png", dpi=200)
    plt.close()


def build_model():
    """Create a compact CNN suitable for 32x32 color images."""
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Rescaling(1.0 / 255),
            augmentation,
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.20),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.30),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(5, activation="softmax"),
        ],
        name="group1_people_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_graph(history):
    """Save the training/validation loss and accuracy curves."""
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history.history["loss"], label="Training loss")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[0].set(title="Training vs. validation loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.history["accuracy"], label="Training accuracy")
    axes[1].plot(
        epochs, history.history["val_accuracy"], label="Validation accuracy"
    )
    axes[1].set(title="Training vs. validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_history.png", dpi=200)
    plt.close(figure)


def save_predictions(model, images, labels):
    """Save ten test images with predicted and true labels."""
    probabilities = model.predict(images[:10], verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    plt.figure(figsize=(12, 5))
    for index in range(10):
        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index])
        color = "green" if predictions[index] == labels[index] else "red"
        plt.title(
            f"P: {CLASS_NAMES[predictions[index]]}\nT: {CLASS_NAMES[labels[index]]}",
            color=color,
            fontsize=9,
        )
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=200)
    plt.close()


def main():
    x_train, y_train, x_test, y_test = load_people_dataset()

    print(f"Training images: {x_train.shape}")  # Expected: (2500, 32, 32, 3)
    print(f"Test images: {x_test.shape}")       # Expected: (500, 32, 32, 3)
    print("Model labels:", dict(enumerate(CLASS_NAMES)))
    print("Training count per class:", np.bincount(y_train))
    print("Test count per class:", np.bincount(y_test))

    save_dataset_examples(x_train, y_train)
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=2,
    )

    save_training_graph(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    save_predictions(model, x_test, y_test)
    model.save(OUTPUT_DIR / "people_classifier.keras")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
