package others;

import java.io.File;

import util.PathUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

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
            
            // First try to load as a Classifier
            Object model = SerializationHelper.read(modelPath);
            
            // Check if it's a Classifier or a Clusterer
            if (model instanceof Classifier) {
                testClassifier((Classifier) model, dataPath);
            } else if (model instanceof Clusterer) {
                testClusterer((Clusterer) model, dataPath);
            } else {
                System.err.println("Unknown model type: " + model.getClass().getName());
            }
            
        } catch (Exception e) {
            System.err.println("Error occurred while testing models:");
            e.printStackTrace();
        }
    }
    
    private static void testClassifier(Classifier classifier, String dataPath) throws Exception {
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
    }
    
    private static void testClusterer(Clusterer clusterer, String dataPath) throws Exception {
        System.out.println("Clusterer loaded successfully: " + clusterer.getClass().getSimpleName());
        
        Instances data = DataSource.read(dataPath);
        int classIndex = data.numAttributes() - 1;
        data.setClassIndex(classIndex);
        
        // Remove class attribute for clustering
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices(String.valueOf(classIndex + 1));  // WEKA is 1-indexed for attributes
        removeFilter.setInvertSelection(false);
        removeFilter.setInputFormat(data);
        Instances dataNoClass = Filter.useFilter(data, removeFilter);
        
        System.out.println("\n=== Cluster Assignments ===");
        System.out.println("Instance, Cluster");
        
        for (int i = 0; i < Math.min(10, dataNoClass.numInstances()); i++) {
            int cluster = clusterer.clusterInstance(dataNoClass.instance(i));
            System.out.printf("%d, %d\n", i+1, cluster);
        }
        
        if (dataNoClass.numInstances() > 10) {
            System.out.println("... (showing first 10 assignments only)");
        }
        
        // Additional evaluations for SimpleKMeans
        if (clusterer instanceof weka.clusterers.SimpleKMeans) {
            weka.clusterers.SimpleKMeans kmeans = (weka.clusterers.SimpleKMeans) clusterer;
            System.out.printf("Within-Cluster Sum of Squared Errors: %.4f\n", kmeans.getSquaredError());
            calculateClusterDistribution(dataNoClass, kmeans);
        }
    }
    
    private static void calculateClusterDistribution(Instances data, weka.clusterers.SimpleKMeans kmeans) throws Exception {
        int numClusters = kmeans.numberOfClusters();
        int[] instancesPerCluster = new int[numClusters];
        
        for (int i = 0; i < data.numInstances(); i++) {
            int clusterNum = kmeans.clusterInstance(data.instance(i));
            instancesPerCluster[clusterNum]++;
        }
        
        System.out.println("\n=== Cluster Distribution ===");
        for (int i = 0; i < numClusters; i++) {
            System.out.printf("Cluster %d: %d instances (%.2f%%)\n", 
                i, instancesPerCluster[i], 
                (double) instancesPerCluster[i] * 100 / data.numInstances());
        }
    }
} 