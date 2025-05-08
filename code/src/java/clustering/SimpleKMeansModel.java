package clustering;

import util.ClusteringRunner;
import weka.clusterers.SimpleKMeans;

public class SimpleKMeansModel {
    public static void main(String[] args) throws Exception {
        SimpleKMeans kMeans = new SimpleKMeans();
        
        int numClusters = 5;
        kMeans.setNumClusters(numClusters);
        
        kMeans.setSeed(1);  // For reproducibility
        kMeans.setMaxIterations(100);
        kMeans.setPreserveInstancesOrder(true);
        
        ClusteringRunner.runClusterer(args, kMeans, numClusters);
    }
}