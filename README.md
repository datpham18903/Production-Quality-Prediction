# Machine Learning Model Framework

A Java-based machine learning framework for quality prediction using the Weka library. This project implements various machine learning algorithms to analyze and predict data patterns.

## Project Structure

```
code/
├── src/
│   ├── java/
│   │   ├── baseline/               # Baseline prediction models
│   │   │   ├── IBkModel.java       # k-nearest neighbor model
│   │   │   ├── LinearRegressionModel.java
│   │   │   ├── M5PModel.java       # M5P decision tree for regression
│   │   │   ├── REPTreeModel.java   # Fast decision tree learner
│   │   │   └── ZeroRModel.java     # Baseline predictor (mean value)
│   │   ├── clustering/             # Clustering & additional models
│   │   │   ├── MultiLayerPerceptronModel.java
│   │   │   ├── RandomForestModel.java
│   │   │   └── SMOregModel.java    # Support vector machine for regression
│   │   ├── ensemble/               # Ensemble methods
│   │   │   ├── BaggingModel.java   # Bagging ensemble
│   │   │   └── StackingModel.java  # Stacking ensemble
│   │   └── util/                   # Utility classes
│   │       ├── ModelLoader.java    # Loads serialized models
│   │       ├── ModelRunner.java    # Common code for running models
│   │       └── PathUtils.java      # File path resolution utilities
│   └── datasets/                   # Training and test datasets
│       ├── train_data.arff
│       └── test_data.arff
├── models/                         # Serialized trained models
│   ├── IBK                         # k-nearest neighbors model file
│   ├── LINEARREGRESSION            # Linear regression model file
│   ├── M5P                         # M5P decision tree model file
│   ├── REPTREE                     # REPTree model file
│   └── ZEROR                       # ZeroR model file
├── pom.xml                         # Maven configuration file
└── run.bat                         # Batch script to run models
```

## Features

1. **Multiple Learning Algorithms**
   - k-nearest neighbors (IBk)
   - Linear Regression
   - M5P decision tree for regression
   - REPTree fast decision tree learner
   - ZeroR (baseline predictor using mean value)
   - Random Forest
   - Support Vector Machine for regression (SMOreg)
   - Multi-Layer Perceptron neural network
   - Ensemble methods (Bagging, Stacking)

2. **Model Persistence**
   - Ability to save trained models
   - Loading pre-trained models for prediction

3. **Command-line Interface**
   - Easy-to-use batch script for running models

## Requirements

- Java 11 or higher
- Maven for dependency management
- Windows environment (for batch script)

## How to Run

1. Ensure you have Java and Maven installed on your system

2. Navigate to the project directory

3. Run a model using the batch file:
   ```
   run.bat [algorithm]
   ```
   
   Available algorithms:
   - IBk - k-nearest neighbor classifier
   - LinearRegression - linear regression model
   - M5P - M5P decision tree for regression
   - REPTree - fast decision tree learner
   - ZeroR - predicts the mean value (baseline)
   - RandomForest - random forest classifier
   - SMOreg - support vector machine for regression
   - Bagging - bagging ensemble method
   - Stacking - stacking ensemble method
   - MLP - multilayer perceptron neural network

4. The model will run with the default dataset configuration

## Examples

Run k-nearest neighbors algorithm:
```
run.bat IBk
```

Run linear regression:
```
run.bat LinearRegression
```

Run M5P decision tree:
```
run.bat M5P
```

## Project Workflow

The main workflow for each model:

1. Load data from ARFF files
2. Prepare and preprocess the data
3. Train the specified model
4. Evaluate the model's performance
5. Display results and metrics

## Extensibility

The framework is designed to be easily extended:

- New models can be added by creating new Java classes following the existing patterns
- Trained models are saved for future use
- Common utilities handle file paths and model execution