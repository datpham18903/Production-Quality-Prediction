package baseline;

import util.ModelRunner;
import weka.classifiers.lazy.IBk;

public class IBkModel {
    public static void main(String[] args) throws Exception {
        IBk ibk = new IBk();
        
        ModelRunner.runModel(args, ibk);
    }
}
