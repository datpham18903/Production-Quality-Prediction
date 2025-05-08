package util;

import java.io.File;

import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class ModelLoader {
    private static final String MODELS_DIR = "models";

    private static String getModelsPath() {
        String workingDir = System.getProperty("user.dir");
        
        if (workingDir.endsWith(PathUtils.CODE_DIR)) {
            return MODELS_DIR;
        } else {
            return PathUtils.CODE_DIR + File.separator + MODELS_DIR;
        }
    }

    public static void classifyWithModel(String modelName, String dataPath) throws Exception {
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
        
        System.out.println("\n=== Applying model to data ===");
        System.out.println("Data file: " + dataPath);
        System.out.println("Number of instances: " + data.numInstances());
        
        System.out.println("\n=== Predictions ===");
        System.out.println("Instance, Actual, Predicted");
        
        for (int i = 0; i < Math.min(10, data.numInstances()); i++) {
            double actual = data.instance(i).classValue();
            double predicted = classifier.classifyInstance(data.instance(i));   
            System.out.printf("%d, %.4f, %.4f\n", i+1, actual, predicted);
        }
        
        if (data.numInstances() > 10) {
            System.out.println("... (showing first 10 predictions only)");
        }
    }

    public static void clusterWithModel(String modelName, String dataPath) throws Exception {
        String modelsPath = getModelsPath();
        String modelPath = modelsPath + File.separator + modelName;
        
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            System.err.println("Clusterer model file does not exist: " + modelPath);
            return;
        }
        
        if (!PathUtils.fileExists(dataPath)) {
            System.err.println("Data file does not exist: " + dataPath);
            return;
        }
        
        System.out.println("Loading clusterer from: " + modelPath);
        Clusterer clusterer = (Clusterer) SerializationHelper.read(modelPath);
        System.out.println("Clusterer loaded successfully: " + clusterer.getClass().getSimpleName());
        
        Instances data = DataSource.read(dataPath);
        
        int classIndex = data.numAttributes() - 1;
        data.setClassIndex(classIndex);
        
        System.out.println("\n=== Applying clusterer to data ===");
        System.out.println("Data file: " + dataPath);
        System.out.println("Number of instances: " + data.numInstances());
        
        System.out.println("\n=== Cluster Assignments ===");
        System.out.println("Instance, Cluster");
        
        for (int i = 0; i < Math.min(10, data.numInstances()); i++) {
            int cluster = clusterer.clusterInstance(data.instance(i));
            System.out.printf("%d, %d\n", i+1, cluster);
        }
        
        if (data.numInstances() > 10) {
            System.out.println("... (showing first 10 assignments only)");
        }
    }
    
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java util.ModelLoader <model-name> <data-path> [type]");
            System.out.println("  model-name: Name of the model file without path (e.g. LINEARREGRESSION)");
            System.out.println("  data-path: Path to the ARFF file to apply the model to");
            System.out.println("  type: Optional. 'classifier' or 'clusterer'. Default is 'classifier'");
            return;
        }
        
        String modelName = args[0];
        String dataPath = args[1];
        String type = args.length > 2 ? args[2].toLowerCase() : "classifier";
        
        if (type.equals("clusterer")) {
            clusterWithModel(modelName, dataPath);
        } else {
            classifyWithModel(modelName, dataPath);
        }
    }
}