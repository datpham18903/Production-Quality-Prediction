import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class RandomForestRun {
    static final String TRAIN_PATH = "data/arff/merged_train.arff";
    static final String TEST_PATH = "data/arff/merged_test.arff";

    public static void main(String[] args) throws Exception {
        Instances train = DataSource.read(TRAIN_PATH);
        Instances test = DataSource.read(TEST_PATH);

        Remove remove = new Remove();
        remove.setAttributeIndices("1");

        remove.setInputFormat(train);
        Instances filteredTrain = Filter.useFilter(train, remove);
        Instances filteredTest = Filter.useFilter(test, remove);

        filteredTrain.setClassIndex(filteredTrain.numAttributes() - 1);
        filteredTest.setClassIndex(filteredTest.numAttributes() - 1);

        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setMaxDepth(0);

        rf.buildClassifier(filteredTrain);
        Evaluation evalLR = new Evaluation(filteredTrain);
        evalLR.evaluateModel(rf, filteredTest);

        System.out.println("----- Random Forest Results -----");
        System.out.println(evalLR.toSummaryString());
    }
}
