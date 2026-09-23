# CIFAR-100 People Classifier - Group 1

This repository contains our Group 1 deep-learning activity using the CIFAR-100 dataset. Our assigned coarse class is **people**, which has five fine classes:

- Baby
- Boy
- Girl
- Man
- Woman

The goal is to train a convolutional neural network (CNN) that receives a small 32 x 32 color image and predicts which of the five classes it belongs to.

## What we did

We made two models so that we could compare an initial attempt with an improved version.

| Model | Purpose | Test accuracy |
| --- | --- | ---: |
| Model 1 | Initial baseline | 21.0% |
| Model 2 | Improved CNN with balanced validation data | 35.2% |

Random guessing would average around 20% because there are five possible classes. Model 2 is better than the baseline, although it is still not a highly accurate real-world classifier. The low resolution of the images and the similarity between the five classes make the task difficult.

## Important files

- `group1_people_cifar100.py` - Model 1 or baseline experiment
- `group1_people_cifar100_v2.py` - Model 2 and the recommended program to run
- `group1_results/` - graphs and sample predictions from Model 1
- `group1_results_v2/` - graphs, predictions, confusion matrix, and metrics from Model 2
- `.gitignore` - prevents the virtual environment and large trained-model files from being uploaded

The generated `.keras` model files are intentionally excluded from GitHub. They can be recreated by running the training program.

## Before running the project

Install the following:

1. Python 3.11
2. Visual Studio Code
3. The Python extension for VS Code
4. Git, if you want to clone or contribute to the repository

Python 3.11 is recommended because it works reliably with the TensorFlow version used in this project.

## Download the repository

Open a terminal and run:

```powershell
git clone https://github.com/sheblink/Grp1_People.git
cd Grp1_People
```

You can also download the repository as a ZIP file from GitHub, but cloning is recommended if you plan to contribute changes.

## Create the virtual environment

Run these commands inside the project folder, one at a time:

```powershell
py -3.11 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should start with `(.venv)` after activation.

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Install the required packages

With `(.venv)` visible in the terminal, run:

```powershell
python -m pip install --upgrade pip
python -m pip install tensorflow matplotlib numpy
```

Verify TensorFlow with:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Run Model 2

Model 2 is our recommended version:

```powershell
python .\group1_people_cifar100_v2.py
```

The first run downloads the CIFAR-100 dataset, which is approximately 169 MB. Training runs on the CPU on native Windows, so the terminal should be left open until the program finishes.

Expected dataset counts:

```text
Training images: (2000, 32, 32, 3)
Validation images: (500, 32, 32, 3)
Test images: (500, 32, 32, 3)
Training count per class: [400 400 400 400 400]
Validation count per class: [100 100 100 100 100]
Test count per class: [100 100 100 100 100]
```

The program may stop before epoch 50 because it uses early stopping. This is intentional: it keeps the model from continuing after validation performance stops improving.

## Model 2 outputs

After training, open the `group1_results_v2` folder. It contains:

- `training_history_v2.png` - training and validation loss/accuracy
- `sample_predictions_v2.png` - examples of correct and incorrect predictions
- `confusion_matrix_v2.png` - errors made for each class
- `metrics_v2.txt` - important numerical results
- `people_classifier_v2.keras` - saved trained model, generated locally and ignored by Git

Our recorded Model 2 results were:

```text
Epochs completed: 30
Best epoch: 20
Best validation accuracy: 38.2%
Test accuracy: 35.2%
```

Small differences between runs are normal because neural-network training includes random operations.

## How Model 2 works

1. CIFAR-100 is loaded using both fine and coarse labels.
2. Only images with the people coarse label (`14`) are selected.
3. The original fine labels are converted to model labels from 0 to 4.
4. Each class contributes 400 training images, 100 validation images, and 100 test images.
5. The CNN learns visual features through convolution and pooling layers.
6. The final softmax layer returns a probability for each of the five classes.
7. The model is evaluated using images that were not used for training.

## Understanding the results

The model is correct when its predicted label matches the true label. Accuracy is calculated as:

```text
number of correct predictions / total number of test images
```

For Model 2:

```text
176 / 500 = 35.2%
```

The training graph also shows some overfitting. Training accuracy continued to rise while validation accuracy started to level off. Early stopping restored the weights from the best validation-loss epoch.

## Working on the project as a group

Before changing anything, get the newest version:

```powershell
git pull
```

Create a separate branch for your change:

```powershell
git switch -c your-name-short-description
```

After making and checking your changes:

```powershell
git add .
git commit -m "Describe what you changed"
git push -u origin your-name-short-description
```

Then open GitHub and create a pull request so the group can review the change before adding it to `main`.

Do not upload `.venv`, `__pycache__`, or generated `.keras` files. They are already covered by `.gitignore`.

## Common issues

### `Activate.ps1` cannot be found

The virtual environment was probably not created, or the terminal is in the wrong folder. Check that the terminal path ends in `Grp1_People`, then create `.venv` again.

### The terminal does not show `(.venv)`

Activate it before installing packages or running the model:

```powershell
.\.venv\Scripts\Activate.ps1
```

### TensorFlow mentions oneDNN or unavailable GPU support

These are informational messages, not errors. The model can train using the CPU.

### Training gives slightly different results

Small differences are expected. Record the actual result produced by your own run rather than copying a number without checking it.

## Current takeaway

This project is mainly an exercise in preparing data, building a CNN, training it, evaluating it honestly, identifying limitations, and improving an initial attempt. Model 2 is not intended to be a production-ready classifier, but it demonstrates the complete deep-learning workflow and performs better than random guessing.
