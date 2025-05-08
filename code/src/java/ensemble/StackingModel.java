package ensemble;

import util.ModelRunner;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Stacking;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;

public class StackingModel {
    
    public static void main(String[] args) throws Exception {
        Stacking stacking = new Stacking();
        
        stacking.setClassifiers(new Classifier[] {
            new LinearRegression(),
            new M5P(),
            new RandomForest()
        });
        
        stacking.setMetaClassifier(new SMOreg());
        
        stacking.setNumFolds(10);
        stacking.setSeed(1);
        
        ModelRunner.runModel(args, stacking);
    }
}