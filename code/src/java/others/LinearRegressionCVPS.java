package others;

import util.ModelRunner;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.SelectedTag;

public class LinearRegressionCVPS {
    
    public static void main(String[] args) throws Exception {
        LinearRegression lr = new LinearRegression();
        lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));
        lr.setEliminateColinearAttributes(false);

        CVParameterSelection cvps = new CVParameterSelection();
        cvps.setClassifier(lr);
        cvps.setNumFolds(10);
        cvps.addCVParameter("R 1.0E-10 1.0E-1 10"); // Ridge parameter range: 1e-10 to 1e-1 in 10 steps
        
        ModelRunner.runModel(args, cvps);
    }
}

