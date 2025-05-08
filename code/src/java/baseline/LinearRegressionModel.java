package baseline;

import util.ModelRunner;
import weka.classifiers.functions.LinearRegression;
import weka.core.SelectedTag;

public class LinearRegressionModel {
    
    public static void main(String[] args) throws Exception {
        LinearRegression lr = new LinearRegression();
        lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));
        
        ModelRunner.runModel(args, lr);
    }
}