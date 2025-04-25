package com.mycompany.weka;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

/**
 * ZeroR model runner with cross-validation and test evaluation
 * @author DB
 */
public class ZeroRrun {
    public static void main(String[] args) throws Exception {

        // Load datasets
        Instances filteredTrain = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\train_data.arff");
        Instances filteredTest = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\test_data.arff");

        // Set class index to the last attribute
        filteredTrain.setClassIndex(filteredTrain.numAttributes() - 1);
        filteredTest.setClassIndex(filteredTest.numAttributes() - 1);

        // === 10-Fold Cross-Validation on Training Data ===
        ZeroR zeroR = new ZeroR();
        Evaluation evalCV = new Evaluation(filteredTrain);
        evalCV.crossValidateModel(zeroR, filteredTrain, 10, new Random(1));
        System.out.println("----- ZeroR 10-Fold Cross-Validation Results on Training Data -----");
        System.out.println(evalCV.toSummaryString());

        // === Train on Full Training Set and Evaluate on Test Set ===
        zeroR.buildClassifier(filteredTrain);
        Evaluation evalTest = new Evaluation(filteredTrain);
        evalTest.evaluateModel(zeroR, filteredTest);
        System.out.println("----- ZeroR Evaluation on Separate Test Data -----");
        System.out.println(evalTest.toSummaryString());
    }
}
