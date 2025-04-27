/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;
import weka.core.SelectedTag;

import java.util.Random;
/**
 *
 * @author DB
 */
public class LRrun {
    public static void main(String[] args) throws Exception {

        
        Instances filteredTrain = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\train_data.arff");
        Instances filteredTest = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\test_data.arff");

        // Set class index (last attribute)
        filteredTrain.setClassIndex(filteredTrain.numAttributes() - 1);
        filteredTest.setClassIndex(filteredTest.numAttributes() - 1);

        // === 10-Fold Cross-Validation on Training Set ===
        LinearRegression lr = new LinearRegression();
        lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));

        Evaluation evalCV = new Evaluation(filteredTrain);
        evalCV.crossValidateModel(lr, filteredTrain, 10, new Random(1));
        System.out.println("----- LR 10-Fold Cross-Validation Results on Training Data -----");
        System.out.println(evalCV.toSummaryString());

        // === Train on Full Training Set and Evaluate on Test Set ===
        lr.buildClassifier(filteredTrain);
        Evaluation evalTest = new Evaluation(filteredTrain);
        evalTest.evaluateModel(lr, filteredTest);
        System.out.println("----- LR Evaluation on Separate Test Data -----");
        System.out.println(evalTest.toSummaryString());
    }
}
