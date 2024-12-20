# Disease Classifier

A deep learning-based project designed to classify diseases based on symptoms. This repository includes scripts for preprocessing datasets, creating optimal data representations, and training a deep neural network model to predict diseases.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Scripts](#files-and-scripts)
- [Model](#model)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview
Disease Classifier is a tool designed to assist in identifying potential diseases based on a user’s symptoms. The system leverages a deep neural network trained on preprocessed symptom-disease data to provide predictions. This project is intended for educational and exploratory purposes.

## Dataset
The project includes the following datasets:
1. `new_df.csv`: A refined dataset prepared for machine learning.
2. `dataset.csv`: The original dataset containing disease-symptom relationships.
3. `final_df.csv`: The processed dataset used for training the classifier.
4. `Symptom-severity.csv`: Details the severity of each symptom.
5. `symptom_Description.csv`: Contains descriptions of symptoms.
6. `symptom_precaution.csv`: Lists precautions to take for each disease.

The `creating an optimal dataset.ipynb` script processes these datasets to produce `final_df.csv`, an optimized representation of the data used for training the classifier.

## Features
- Preprocesses and cleans raw datasets by standardizing symptoms and removing inconsistencies.
- Construct a deep learning model using TensorFlow and Keras to classify diseases.
- Output predictions based on symptom input, with the top prediction highlighted.

## Requirements
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`
- `joblib`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/disease-classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd disease-classifier
   ```
3. Install dependencies manually using pip:
   ```bash
   pip install pandas numpy scikit-learn tensorflow keras joblib
   ```

## Usage
### Step 1: Preprocess Data
Run the `creating an optimal dataset.ipynb` script to preprocess and generate the `final_df.csv` dataset:
- Cleans the original dataset by standardizing and encoding symptom names.
- Saves the processed dataset for training the model.

### Step 2: Train the Model
Execute the `disease_clf_deepLearning.ipynb` notebook to:
- Load the preprocessed dataset.
- Train a deep learning model on symptom-disease data.
- Save the trained model for later use (optional).

### Step 3: Predict Diseases
Modify the prediction block in `disease_clf_deepLearning.ipynb` to test new symptom inputs. The model outputs the most probable disease based on the given symptoms.

## Files and Scripts
- **`creating an optimal dataset.ipynb`**: Preprocesses datasets to create `final_df.csv`.
- **`disease_clf_deepLearning.ipynb`**: Builds and trains a deep learning model, and makes predictions.
- **`Symptom-severity.csv`**: Provides information on the severity of each symptom.
- **`dataset.csv`**: The original dataset containing disease-symptom relationships.
- **`final_df.csv`**: The processed dataset used for training the classifier.
- **`symptom_Description.csv`**: Contains descriptions of symptoms.
- **`symptom_precaution.csv`**: Lists precautions for each disease.

## Model
The neural network model consists of:
- **Input Layer**: Accepts 131 symptom features.
- **Hidden Layers**:
  - Dense layer with 132 nodes and sigmoid activation.
  - Dense layer with 50 nodes and ReLU activation.
  - Dense layer with 17 nodes and ReLU activation.
- **Output Layer**: Dense layer with 41 nodes and sigmoid activation, representing the number of diseases.

The model is compiled with:
- **Optimizer**: Adam
- **Loss Function**: Binary cross-entropy
- **Evaluation Metric**: Accuracy

## Future Work
- Improve model accuracy by experimenting with hyperparameters and architectures.
- Add support for multi-class disease predictions.
- Develop a user-friendly interface (e.g., web or mobile app) for interacting with the classifier.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bugs, improvements, or feature requests.
