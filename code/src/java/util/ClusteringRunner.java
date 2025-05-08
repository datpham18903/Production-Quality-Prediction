package util;

import java.io.File;

import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ClusteringRunner {
    private static final String MODELS_DIR = "models";

    public static void runClusterer(String[] args, Clusterer clusterer, int numClusters) throws Exception {
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
        
        int classIndex = train.numAttributes() - 1;
        train.setClassIndex(classIndex);
        test.setClassIndex(classIndex);
        
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices(String.valueOf(classIndex + 1));  // WEKA is 1-indexed for attributes
        removeFilter.setInvertSelection(false);
        removeFilter.setInputFormat(train);
        
        Instances trainNoClass = Filter.useFilter(train, removeFilter);
        Instances testNoClass = Filter.useFilter(test, removeFilter);
        
        long startTime = System.currentTimeMillis();
        clusterer.buildClusterer(trainNoClass);
        long endTime = System.currentTimeMillis();
        double buildTime = (endTime - startTime) / 1000.0;
        
        System.out.println("=== Clustering Model ===\n");
        System.out.println(clusterer.toString());
        System.out.printf("Time taken to build model: %.2f seconds\n", buildTime);
        
        saveClustererToFile(clusterer);
        
        evaluateClustering(clusterer, trainNoClass, numClusters, "training");
        evaluateClustering(clusterer, testNoClass, numClusters, "test");
    }

    private static void saveClustererToFile(Clusterer clusterer) throws Exception {
        String clustererName = clusterer.getClass().getSimpleName();
        
        if (clustererName.endsWith("Model")) {
            clustererName = clustererName.substring(0, clustererName.length() - 5);
        }
        
        String fileName = clustererName.toUpperCase();
        
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
        System.out.println("Saving clusterer to binary file: " + filePath);
        SerializationHelper.write(filePath, clusterer);
        System.out.println("Clusterer saved successfully to: " + new File(filePath).getAbsolutePath());
    }

    private static void evaluateClustering(Clusterer clusterer, Instances data, int numClusters, String datasetName) throws Exception {
        System.out.println("\n=== Clustering evaluation on " + datasetName + " data ===");
        
        int[] instancesPerCluster = new int[numClusters];
        for (int i = 0; i < data.numInstances(); i++) {
            int clusterNum = clusterer.clusterInstance(data.instance(i));
            instancesPerCluster[clusterNum]++;
        }
        
        System.out.println("Number of clusters: " + numClusters);
        for (int i = 0; i < numClusters; i++) {
            System.out.printf("Cluster %d: %d instances (%.2f%%)\n", 
                i, instancesPerCluster[i], 
                (double) instancesPerCluster[i] * 100 / data.numInstances());
        }
        
        if (clusterer instanceof weka.clusterers.SimpleKMeans) {
            weka.clusterers.SimpleKMeans kmeans = (weka.clusterers.SimpleKMeans) clusterer;
            System.out.printf("Within-Cluster Sum of Squared Errors: %.4f\n", kmeans.getSquaredError());
        }
        
        if (clusterer instanceof weka.clusterers.SimpleKMeans) {
            weka.clusterers.SimpleKMeans kmeans = (weka.clusterers.SimpleKMeans) clusterer;
            calculateAverageDistanceToCentroid(data, kmeans);
        }
    }
    
    private static void calculateAverageDistanceToCentroid(Instances data, weka.clusterers.SimpleKMeans kmeans) throws Exception {
        double totalDistance = 0;
        Instances centroids = kmeans.getClusterCentroids();
        
        for (int i = 0; i < data.numInstances(); i++) {
            int clusterIndex = kmeans.clusterInstance(data.instance(i));
            double distance = calculateEuclideanDistance(data.instance(i), centroids.instance(clusterIndex));
            totalDistance += distance;
        }
        
        double avgDistance = totalDistance / data.numInstances();
        System.out.printf("Average distance to centroid: %.4f\n", avgDistance);
    }
    
    private static double calculateEuclideanDistance(weka.core.Instance instance1, weka.core.Instance instance2) {
        double sum = 0;
        for (int i = 0; i < instance1.numAttributes(); i++) {
            double diff = instance1.value(i) - instance2.value(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}