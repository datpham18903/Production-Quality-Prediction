package baseline;

import util.ModelRunner;
import weka.classifiers.rules.ZeroR;

public class ZeroRModel {
    
    public static void main(String[] args) throws Exception {
        ZeroR zeroR = new ZeroR();
        
        ModelRunner.runModel(args, zeroR);
    }
}
