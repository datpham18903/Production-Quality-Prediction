package com.mycompany.weka;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

/**
 * M5P model runner with cross-validation and test evaluation
 * @author DB
 */
public class M5Prun {
    public static void main(String[] args) throws Exception {

        // Load datasets
        Instances filteredTrain = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\train_data.arff");
        Instances filteredTest = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\test_data.arff");

        // Set class index to the last attribute
        filteredTrain.setClassIndex(filteredTrain.numAttributes() - 1);
        filteredTest.setClassIndex(filteredTest.numAttributes() - 1);

        // === 10-Fold Cross-Validation on Training Data ===
        M5P m5p = new M5P();
        Evaluation evalCV = new Evaluation(filteredTrain);
        evalCV.crossValidateModel(m5p, filteredTrain, 10, new Random(1));
        System.out.println("----- M5P 10-Fold Cross-Validation Results on Training Data -----");
        System.out.println(evalCV.toSummaryString());

        // === Train on Full Training Set and Evaluate on Test Set ===
        m5p.buildClassifier(filteredTrain);
        Evaluation evalTest = new Evaluation(filteredTrain);
        evalTest.evaluateModel(m5p, filteredTest);
        System.out.println("----- M5P Evaluation on Separate Test Data -----");
        System.out.println(evalTest.toSummaryString());
    }
}
