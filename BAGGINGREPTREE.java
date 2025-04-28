
/**
 * @kevinphuc
 * @params train_data.arff, test_data.arff
 * @description This program evaluates the performance of a Bagging model using REPTree as the base classifier.
 * * It performs 10-fold cross-validation on the training set and evaluates the model on a separate test set.
 * * It prints out various evaluation metrics such as correlation coefficient, mean absolute error, root mean squared error,
 * * * relative absolute error, and root relative squared error.
 * * * The program uses Weka library for machine learning tasks.
 * * * The model is built using the Bagging algorithm with REPTree as the base classifier.
 * * * The program reads the training and test data from ARFF files.
 * * * * The model is trained on the training set and evaluated on the test set.
 * * * * The program prints out the evaluation metrics for both the cross-validation and test set evaluations.
 */

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.REPTree;
import weka.classifiers.meta.Bagging;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Random;

public class BAGGINGREPTREE {

    public static Classifier buildModel() throws Exception {
        Bagging bag = new Bagging();
        REPTree tree = new REPTree();
        bag.setClassifier(tree);
        bag.setNumIterations(25);
        bag.setSeed(1);
        return bag;
    }

    private static void printStats(Evaluation eval) throws Exception {
        System.out.printf("Correlation coefficient: %.4f%n", eval.correlationCoefficient());
        System.out.printf("Mean absolute error: %.4f%n", eval.meanAbsoluteError());
        System.out.printf("Root mean squared error: %.4f%n", eval.rootMeanSquaredError());
        System.out.printf("Relative absolute error: %.2f%%%n", eval.relativeAbsoluteError());
        System.out.printf("Root relative squared error: %.2f%%%n", eval.rootRelativeSquaredError());
    }

    public static void main(String[] args) throws Exception {
        Instances train = DataSource.read("train_data.arff");
        Instances test = DataSource.read("test_data.arff");

        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);
        if (test.classIndex() == -1)
            test.setClassIndex(test.numAttributes() - 1);

        Classifier model = buildModel();

        System.out.println("===== Bagging(REPTree) =====");

        // 1) 10-fold CV on train
        Evaluation evalCV = new Evaluation(train);
        evalCV.crossValidateModel(model, train, 10, new Random(1));
        System.out.println("-- Cross-validation on training set --");
        printStats(evalCV);

        // 2) Train full on train, evaluate on test
        Evaluation evalTest = new Evaluation(train);
        model.buildClassifier(train);
        evalTest.evaluateModel(model, test);
        System.out.println("-- Evaluation on separate test set --");
        printStats(evalTest);
    }
}
