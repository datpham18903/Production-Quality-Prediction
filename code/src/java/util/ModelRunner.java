package util;

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class ModelRunner {
    
    private static final String MODELS_DIR = "models";

    public static void runModel(String[] args, Classifier classifier) throws Exception {
        String[] paths = PathUtils.resolveDataPaths(args);
        String trainPath = paths[0];
        String testPath = paths[1];
        
        System.out.println("Using training data: " + trainPath);
        System.out.println("Using testing data: " + testPath);
        
        if (!PathUtils.fileExists(trainPath)) {
            System.err.println("Training file does not exist: " + trainPath);
            return;
        }
        
        if (!PathUtils.fileExists(testPath)) {
            System.err.println("Testing file does not exist: " + testPath);
            return;
        }
        
        Instances train = DataSource.read(trainPath);
        Instances test = DataSource.read(testPath);
        
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
        
        long startTime = System.currentTimeMillis();
        classifier.buildClassifier(train);
        System.out.println("=== Classifier model (full training set) ===");
        System.out.println(classifier.toString());
        long endTime = System.currentTimeMillis();
        double buildTime = (endTime - startTime) / 1000.0;
        System.out.printf("Time taken to build model: %.2f seconds\n", buildTime);
        
        saveModelToFile(classifier);
        
        startTime = System.currentTimeMillis();
        Evaluation evalCV = new Evaluation(train);
        evalCV.crossValidateModel(classifier, train, 10, new Random(1));
        System.out.println("\n=== Cross-validation ===");
        System.out.println("=== Summary ===");
        System.out.println(evalCV.toSummaryString());
        
        Evaluation evalTest = new Evaluation(train);
        evalTest.evaluateModel(classifier, test);
        System.out.println("\n=== Evaluation (testing data) ===");
        System.out.println(evalTest.toSummaryString());
        endTime = System.currentTimeMillis();
        double evalTime = (endTime - startTime) / 1000.0;
        System.out.printf("Time taken to evaluate model: %.2f seconds\n", evalTime);
    }
    
    private static void saveModelToFile(Classifier classifier) throws Exception {
        String classifierName = classifier.getClass().getSimpleName();
        
        if (classifierName.endsWith("Model")) {
            classifierName = classifierName.substring(0, classifierName.length() - 5);
        }
        
        String fileName = classifierName.toUpperCase();
        
        String workingDir = System.getProperty("user.dir");
        String modelsPath;
        
        if (workingDir.endsWith(PathUtils.CODE_DIR)) {
            modelsPath = MODELS_DIR;
        } else {
            modelsPath = PathUtils.CODE_DIR + File.separator + MODELS_DIR;
        }
        
        File outputDir = new File(modelsPath);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        
        String filePath = modelsPath + File.separator + fileName;
        System.out.println("Saving model to binary file: " + filePath);
        SerializationHelper.write(filePath, classifier);
        System.out.println("Model saved successfully to: " + new File(filePath).getAbsolutePath());
    }
    
    public static void main(String[] args) throws Exception {
        System.out.println("ModelRunner example usage:");
        System.out.println("Command line arguments: " + (args.length > 0 ? String.join(", ", args) : "none"));
        System.out.println("Default train path: " + PathUtils.DEFAULT_TRAIN_PATH);
        System.out.println("Default test path: " + PathUtils.DEFAULT_TEST_PATH);
        
        System.out.println("\nTo use this utility class with any model, add the following to your model class:");
        System.out.println("import util.ModelRunner;");
        System.out.println("import weka.classifiers.trees.RandomForest; // Your classifier");
        System.out.println("\n// In your main method:");
        System.out.println("RandomForest classifier = new RandomForest(); // Initialize your classifier");
        System.out.println("// Configure classifier parameters as needed");
        System.out.println("ModelRunner.runModel(args, classifier); // Run the model with path handling");
    }
}