# Production Quality Prediction - A Machine Learning Model Framework

A Java-based machine learning framework for production quality prediction using the Weka library. This project implements various machine learning algorithms to analyze and predict data patterns.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/datpham18903/Production-Quality-Prediction.git
   ```

2. Install dependencies using Maven:
   ```bash
   cd Production-Quality-Prediction/code
   mvn clean install
   ```

## Dependencies

- Java 11 or higher
- Maven 3.6 or higher
- Weka 3.8.6
- JFreeChart 1.5.3
- LibSVM 3.24
- MTJ 1.0.4
- Other dependencies are listed in `pom.xml`

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

## How to Run

1. Ensure you have Java and Maven installed on your system

2. Navigate to the project directory

3. Run a model using one of the following commands:

   **For PowerShell:**
   ```
   .\run.bat [algorithm]
   ```

   **For Command Prompt:**
   ```
   run.bat [algorithm]
   ```
   
   Available algorithms:
   - IBk - k-Nearest Neighbor Classifier
   - LinearRegression - Linear Regression Model
   - M5P - M5P Decision Tree for Regression
   - REPTree - Fast Decision Tree Learner
   - ZeroR - Predicts the Mean Value (Baseline)
   - RandomForest - Random Forest Classifier
   - SMOreg - Support Vector Machine for Regression
   - Bagging - Bagging Ensemble Method
   - Stacking - Stacking Ensemble Method
   - MLP - Multilayer Perceptron Neural Network
   - SimpleKMeans - K-Means Clustering

4. The model will run with the default dataset configuration

## Examples

Run k-nearest neighbors algorithm:
```
.\run.bat IBk     # For PowerShell
run.bat IBk       # For Command Prompt
```

Run linear regression:
```
.\run.bat LinearRegression     # For PowerShell
run.bat LinearRegression       # For Command Prompt
```

Run M5P decision tree:
```
.\run.bat M5P     # For PowerShell
run.bat M5P       # For Command Prompt
```

## Project Workflow

The main workflow for each model:

1. Load data from ARFF files
2. Prepare and preprocess the data
3. Train the specified model
4. Evaluate the model's performance
5. Display results and metrics

## Troubleshooting

Common issues and solutions:

1. **OutOfMemoryError**
   - Increase JVM heap size in run.bat: `java -Xmx4g -jar ...`
   - Reduce dataset size or use data sampling

2. **Model not found error**
   - Ensure model files exist in the models/ directory
   - Try retraining the model
   - Check file permissions

3. **Java version mismatch**
   - Ensure you're using Java 11 or higher
   - Check JAVA_HOME environment variable

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Adding New Models

To add a new model:

1. Create a new class in appropriate package (baseline/clustering/ensemble)
2. Implement the required model interface
3. Add model to run.bat script
4. Update documentation
5. Add tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Weka library developers
- Contributors and maintainers
- [List any other acknowledgments]