"""Model 2 for Group 1: CIFAR-100 people classification.

This version keeps the original experiment as a baseline and tests a revised
custom CNN with a balanced validation set and gentler regularization.
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
EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_IMAGES_PER_CLASS = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent / "group1_results_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_people_dataset():
    """Load CIFAR-100 and retain only the five classes under people."""
    (train_images, train_fine), (test_images, test_fine) = (
        keras.datasets.cifar100.load_data(label_mode="fine")
    )
    (_, train_coarse), (_, test_coarse) = keras.datasets.cifar100.load_data(
        label_mode="coarse"
    )

    train_fine = train_fine.squeeze()
    test_fine = test_fine.squeeze()
    train_coarse = train_coarse.squeeze()
    test_coarse = test_coarse.squeeze()

    train_mask = train_coarse == PEOPLE_COARSE_ID
    test_mask = test_coarse == PEOPLE_COARSE_ID

    x_all = train_images[train_mask]
    y_all_original = train_fine[train_mask]
    x_test = test_images[test_mask]
    y_test_original = test_fine[test_mask]

    label_lookup = np.full(100, -1, dtype=np.int32)
    label_lookup[ORIGINAL_FINE_IDS] = np.arange(len(CLASS_NAMES))
    y_all = label_lookup[y_all_original]
    y_test = label_lookup[y_test_original]

    if np.any(y_all < 0) or np.any(y_test < 0):
        raise ValueError("Unexpected fine label found in the people superclass.")

    return x_all, y_all, x_test, y_test


def balanced_train_validation_split(images, labels):
    """Reserve exactly 100 validation images from every class."""
    rng = np.random.default_rng(SEED)
    training_indices = []
    validation_indices = []

    for class_id in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        validation_indices.extend(class_indices[:VALIDATION_IMAGES_PER_CLASS])
        training_indices.extend(class_indices[VALIDATION_IMAGES_PER_CLASS:])

    training_indices = np.asarray(training_indices)
    validation_indices = np.asarray(validation_indices)
    rng.shuffle(training_indices)
    rng.shuffle(validation_indices)

    return (
        images[training_indices],
        labels[training_indices],
        images[validation_indices],
        labels[validation_indices],
    )


def build_model():
    """Build a custom CNN that retains more spatial information."""
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        ],
        name="light_augmentation",
    )

    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Rescaling(1.0 / 255),
            augmentation,
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.10),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.20),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.30),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(5, activation="softmax"),
        ],
        name="group1_people_classifier_v2",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_graph(history):
    """Save loss and accuracy curves for the report."""
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history.history["loss"], label="Training loss")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[0].set(title="Model 2: loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.history["accuracy"], label="Training accuracy")
    axes[1].plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    axes[1].set(title="Model 2: accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_history_v2.png", dpi=200)
    plt.close(figure)


def save_predictions_and_confusion_matrix(model, images, labels):
    """Save visual predictions and a five-class confusion matrix."""
    probabilities = model.predict(images, verbose=0)
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
    plt.savefig(OUTPUT_DIR / "sample_predictions_v2.png", dpi=200)
    plt.close()

    matrix = tf.math.confusion_matrix(labels, predictions, num_classes=5).numpy()
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Model 2 confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(5),
        yticks=range(5),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    for row in range(5):
        for column in range(5):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix_v2.png", dpi=200)
    plt.close(figure)


def save_metrics(history, test_loss, test_accuracy):
    """Save the important numbers in a small report-friendly text file."""
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    best_val_accuracy = float(history.history["val_accuracy"][best_epoch - 1])

    text = (
        "MODEL 2 RESULTS\n"
        f"Epochs completed: {len(history.history['loss'])}\n"
        f"Best epoch by validation loss: {best_epoch}\n"
        f"Validation loss at best epoch: {best_val_loss:.4f}\n"
        f"Validation accuracy at best epoch: {best_val_accuracy:.4f}\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n"
    )
    (OUTPUT_DIR / "metrics_v2.txt").write_text(text, encoding="utf-8")


def main():
    x_all, y_all, x_test, y_test = load_people_dataset()
    x_train, y_train, x_val, y_val = balanced_train_validation_split(x_all, y_all)

    print(f"Training images: {x_train.shape}")
    print(f"Validation images: {x_val.shape}")
    print(f"Test images: {x_test.shape}")
    print("Training count per class:", np.bincount(y_train))
    print("Validation count per class:", np.bincount(y_val))
    print("Test count per class:", np.bincount(y_test))

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    save_training_graph(history)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    save_predictions_and_confusion_matrix(model, x_test, y_test)
    save_metrics(history, test_loss, test_accuracy)
    model.save(OUTPUT_DIR / "people_classifier_v2.keras")
    print(f"Saved Model 2 results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
