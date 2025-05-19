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
- Python 3.x 

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
│   │   ├── others/                 # Additional model implementations & testing
│   │   │   ├── MultiLayerPerceptronModel.java
│   │   │   ├── RandomForestModel.java
│   │   │   ├── SMOregModel.java
│   │   │   └── SampleTesting.java  # Sample testing implementation
│   │   └── util/                   # Utility classes
│   │       ├── ModelLoader.java    # Loads serialized models
│   │       ├── ModelRunner.java    # Common code for running models
│   │       └── PathUtils.java      # File path resolution utilities
│   ├── python/                     # Python scripts for data preprocessing
│   │   └── preprocessing.ipynb     # Jupyter notebook for data preprocessing
│   └── datasets/                   # Training and test datasets
│       ├── train_data.arff
│       └── test_data.arff
├── models/                         # Serialized trained models
│   ├── IBK                         # k-nearest neighbors model file
│   ├── LINEARREGRESSION            # Linear regression model file
│   ├── M5P                         # M5P decision tree model file
│   ├── REPTREE                     # REPTree model file
│   └── ZEROR                       # ZeroR model file
├── lib/                            # Additional libraries
├── pom.xml                         # Maven configuration file
├── run.bat                         # Batch script to run models
└── test.bat                        # Batch script for testing models
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
   - Serialized model storage for efficient reuse

3. **Command-line Interface**
   - Easy-to-use batch scripts for running and testing models

4. **Data Preprocessing**
   - Python-based data preprocessing capabilities
   - Jupyter notebook for interactive data exploration and transformation

## Model Size Limitations

Due to file size limitations, some larger serialized model files (BAGGING, STACKING, and RANDOMFOREST) are not included in the repository. The source code for these models (BaggingModel.java, StackingModel.java, and RandomForestModel.java) is still available in the codebase, and you can:

1. Train these models locally using the provided source code
2. Use model compression techniques for the serialized files
3. Consider using model quantization or pruning to reduce model size

Note: While the serialized model files are not included, all source code files are maintained in the repository, allowing you to train and generate these models locally.

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

4. For testing specific models, use the test script:

   **For PowerShell:**
   ```
   .\test.bat [model]
   ```

   **For Command Prompt:**
   ```
   test.bat [model]
   ```
   
   Available models:
   - IBK - k-Nearest Neighbor Classifier
   - LINEARREGRESSION - Linear Regression Model
   - M5P - M5P Decision Tree for Regression
   - REPTREE - Fast Decision Tree Learner
   - ZEROR - Predicts the Mean Value (Baseline)
   - RANDOMFOREST - Random Forest Classifier
   - SMOREG - Support Vector Machine for Regression
   - BAGGING - Bagging Ensemble Method
   - STACKING - Stacking Ensemble Method
   - MLP - Multilayer Perceptron Neural Network
   - SIMPLEKMEANS - K-Means Clustering (Clusterer type)

5. The model will run with the default dataset configuration

## Project Workflow

The main workflow for each model:

1. Preprocess data using Python scripts (preprocessing.ipynb)
2. Load data from ARFF files
3. Prepare and preprocess the data
4. Train the specified model
5. Evaluate the model's performance
6. Display results and metrics
7. Save model for future use

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

4. **Python dependency issues**
   - Install required Python packages for data preprocessing: `pip install jupyter pandas numpy scikit-learn matplotlib seaborn liac-arff`
   - Ensure correct Python version is installed

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
- Contributors to the scikit-learn and pandas libraries