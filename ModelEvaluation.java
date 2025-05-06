import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.trees.REPTree;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SGD;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.SelectedTag;

import java.util.Random;

/**
 * ModelEvaluation
 *
 * Loads ARFF training and test datasets and evaluates several regression models
 * using 10-fold cross-validation on the training set, then trains on the full
 * training set and evaluates on the separate test set.
 *
 * Usage:
 *   java -cp ".:weka.jar" ModelEvaluation train_data.arff test_data.arff
 */
public class ModelEvaluation {
    public static void main(String[] args) throws Exception {
        // Load train and test sets
        Instances train = DataSource.read("train_data.arff");
        Instances test = DataSource.read("test_data.arff");
        if (train.classIndex() == -1) train.setClassIndex(train.numAttributes() - 1);
        if (test.classIndex()  == -1) test.setClassIndex(test.numAttributes() - 1);

        // Define models
        Classifier[] models = {
            getAdditiveRegression(),
            getBagging(),
            getStacking(),
            getSGD()
        };
        String[] names = {
            "AdditiveRegression",
            "Bagging(REPTree)",
            "Stacking(LR+REPTree)",
            "SGD"
        };

        // Evaluate each model
        for (int i = 0; i < models.length; i++) {
            System.out.println("===== " + names[i] + " =====");

            // 1) 10-fold CV on train
            Evaluation evalCV = new Evaluation(train);
            evalCV.crossValidateModel(models[i], train, 10, new Random(1));
            System.out.println("-- Cross-validation on training set --");
            printStats(evalCV);

            // 2) Train full on train, evaluate on test
            Evaluation evalTest = new Evaluation(train);
            models[i].buildClassifier(train);
            evalTest.evaluateModel(models[i], test);
            System.out.println("-- Evaluation on separate test set --");
            printStats(evalTest);

            System.out.println();
        }
    }

    private static void printStats(Evaluation eval) throws Exception {
        System.out.printf("Correlation coefficient: %.4f%n", eval.correlationCoefficient());
        System.out.printf("Mean absolute error: %.4f%n", eval.meanAbsoluteError());
        System.out.printf("Root mean squared error: %.4f%n", eval.rootMeanSquaredError());
        System.out.printf("Relative absolute error: %.2f%%%n", eval.relativeAbsoluteError());
        System.out.printf("Root relative squared error: %.2f%%%n", eval.rootRelativeSquaredError());
    }

    private static Classifier getAdditiveRegression() throws Exception {
        AdditiveRegression ar = new AdditiveRegression();
        REPTree tree = new REPTree();
        tree.setMaxDepth(3);
        tree.setMinNum(2);
        ar.setClassifier(tree);
        ar.setNumIterations(75);
        // ar.setSeed(1);
        return ar;
    }

    private static Classifier getBagging() throws Exception {
        Bagging bag = new Bagging();
        REPTree tree = new REPTree();
        bag.setClassifier(tree);
        bag.setNumIterations(25);
        bag.setSeed(1);
        return bag;
    }

    private static Classifier getStacking() throws Exception {
        Stacking stack = new Stacking();
        Classifier[] base = {
            new LinearRegression(),
            new REPTree()
        };
        stack.setClassifiers(base);
        stack.setMetaClassifier(new LinearRegression());
        return stack;
    }

    private static Classifier getSGD() throws Exception {
        SGD sgd = new SGD();
        sgd.setLossFunction(new SelectedTag(SGD.SQUAREDLOSS, SGD.TAGS_SELECTION));
        sgd.setLearningRate(0.0001);
        sgd.setEpochs(100);
        sgd.setSeed(1);
        return sgd;
    }
}
