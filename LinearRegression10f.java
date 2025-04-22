/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.weka;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.SelectedTag;

import java.util.Random;
/**
 *
 * @author DB
 */
public class LinearRegression10f {
    public static void main(String[] args) throws Exception {

        Instances train = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\merged_train.arff");

        Remove remove = new Remove();
        remove.setAttributeIndices("1");
        remove.setInputFormat(train);
        Instances filteredTrain = Filter.useFilter(train, remove);

        filteredTrain.setClassIndex(filteredTrain.numAttributes() - 1);

        LinearRegression lr = new LinearRegression();
        lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));

        Evaluation evalCV = new Evaluation(filteredTrain);
        evalCV.crossValidateModel(lr, filteredTrain, 10, new Random(1));
        System.out.println("----- 10-Fold Cross-Validation on Training Data -----");
        System.out.println(evalCV.toSummaryString());


    }
}
