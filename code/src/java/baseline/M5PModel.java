package baseline;

import util.ModelRunner;
import weka.classifiers.trees.M5P;

public class M5PModel {
    
    public static void main(String[] args) throws Exception {
        M5P m5p = new M5P();
        
        ModelRunner.runModel(args, m5p);
    }
}
