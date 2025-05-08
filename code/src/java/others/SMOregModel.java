package others;

import util.ModelRunner;
import weka.classifiers.functions.SMOreg;

public class SMOregModel {
    
    public static void main(String[] args) throws Exception {
        SMOreg smoReg = new SMOreg();
        
        ModelRunner.runModel(args, smoReg);
    }
}
