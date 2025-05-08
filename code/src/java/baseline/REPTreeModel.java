package baseline;

import util.ModelRunner;
import weka.classifiers.trees.REPTree;

public class REPTreeModel {
    
    public static void main(String[] args) throws Exception {
        REPTree repTree = new REPTree();
        
        ModelRunner.runModel(args, repTree);
    }
}