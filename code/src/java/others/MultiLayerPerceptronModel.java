package others;

import util.ModelRunner;
import weka.classifiers.functions.MultilayerPerceptron;

public class MultiLayerPerceptronModel {
    
    public static void main(String[] args) throws Exception {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        
        ModelRunner.runModel(args, mlp);
    }
}
