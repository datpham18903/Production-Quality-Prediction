package ensemble;

import util.ModelRunner;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;

public class BaggingModel {
    
    public static void main(String[] args) throws Exception {
        RandomForest rf = new RandomForest();
        rf.setMaxDepth(0);
        rf.setNumFeatures(0);
        rf.setNumIterations(10);
        
        Bagging bagging = new Bagging();
        bagging.setClassifier(rf);
        bagging.setNumIterations(20);
        bagging.setBagSizePercent(100);
        bagging.setCalcOutOfBag(true);
        bagging.setSeed(1);
        
        ModelRunner.runModel(args, bagging);
    }
}