package others;

import util.ModelRunner;
import weka.classifiers.trees.RandomForest;

public class RandomForestModel {
    
    public static void main(String[] args) throws Exception {
        RandomForest randomForest = new RandomForest();
        randomForest.setNumIterations(100);
        randomForest.setMaxDepth(0);
        randomForest.setSeed(1);
        
        ModelRunner.runModel(args, randomForest);
    }
}