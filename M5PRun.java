/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.weka;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author DB
 */
public class M5PRun {
    public static void main(String[] args) throws Exception {

        Instances train = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\merged_train.arff");
        Instances test = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\merged_test.arff");

        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

        Remove remove = new Remove();
        remove.setAttributeIndices("1");
        remove.setInputFormat(train);
        Instances filteredTrain = Filter.useFilter(train, remove);
        Instances filteredTest = Filter.useFilter(test, remove);

        System.out.println("----- M5P Results -----");
        LibSVM svm = new LibSVM();
        svm.buildClassifier(filteredTrain);

        Evaluation evalSVM = new Evaluation(filteredTrain);
        evalSVM.evaluateModel(svm, filteredTest);
        System.out.println(evalSVM.toSummaryString());
        
    }
}
