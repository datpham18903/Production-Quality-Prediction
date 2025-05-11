package others;

import java.io.File;

import util.PathUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class SampleTesting {
    private static final String MODELS_DIR = "models";

    private static String getModelsPath() {
        String workingDir = System.getProperty("user.dir");
        
        if (workingDir.endsWith(PathUtils.CODE_DIR)) {
            return MODELS_DIR;
        } else {
            return PathUtils.CODE_DIR + File.separator + MODELS_DIR;
        }
    }
    
    public static void main(String[] args) {
        try {
            String modelName = "LINEARREGRESSION";
            
            if (args.length > 0 && args[0] != null && !args[0].isEmpty()) {
                modelName = args[0].toUpperCase();
            }
            
            String dataPath = "src/datasets/sample_data.arff";
            
            System.out.println("\n===== TESTING " + modelName + " MODEL =====");
            
            String modelsPath = getModelsPath();
            String modelPath = modelsPath + File.separator + modelName;
            
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                System.err.println("Model file does not exist: " + modelPath);
                return;
            }
            
            if (!PathUtils.fileExists(dataPath)) {
                System.err.println("Data file does not exist: " + dataPath);
                return;
            }
            
            System.out.println("Loading model from: " + modelPath);
            Classifier classifier = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("Model loaded successfully: " + classifier.getClass().getSimpleName());
            
            Instances data = DataSource.read(dataPath);
            data.setClassIndex(data.numAttributes() - 1);
            
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(classifier, data);
            
            System.out.println("\n=== Sample Predictions ===");
            System.out.println("Instance, Actual, Predicted");
            
            for (int i = 0; i < Math.min(5, data.numInstances()); i++) {
                double actual = data.instance(i).classValue();
                double predicted = classifier.classifyInstance(data.instance(i));   
                System.out.printf("%d, %.4f, %.4f\n", i+1, actual, predicted);
            }
            
            System.out.println("\n=== Evaluation Results ===");
            System.out.printf("Correlation coefficient: %.4f\n", eval.correlationCoefficient());
            System.out.printf("Mean absolute error: %.4f\n", eval.meanAbsoluteError());
            System.out.printf("Root mean squared error: %.4f\n", eval.rootMeanSquaredError());
            
            // Handle infinity values for relative errors
            double relAbsError = eval.relativeAbsoluteError();
            if (Double.isInfinite(relAbsError)) {
                System.out.println("Relative absolute error: N/A (Zero variance in actual values)");
            } else {
                System.out.printf("Relative absolute error: %.4f%%\n", relAbsError);
            }
            
            double rootRelError = eval.rootRelativeSquaredError();
            if (Double.isInfinite(rootRelError)) {
                System.out.println("Root relative squared error: N/A (Zero variance in actual values)");
            } else {
                System.out.printf("Root relative squared error: %.4f%%\n", rootRelError);
            }
            
            System.out.printf("Total Number of Instances: %.4f\n", eval.numInstances());
            
        } catch (Exception e) {
            System.err.println("Error occurred while testing models:");
            e.printStackTrace();
        }
    }
} 