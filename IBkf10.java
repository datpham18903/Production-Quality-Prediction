/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.weka;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author DB
 */
public class IBkf10 {
    public static void main(String[] args) throws Exception {

        Instances train = ConverterUtils.DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\data_compressed.arff");

        train.setClassIndex(train.numAttributes() - 1);

        IBk lr = new IBk();

        Evaluation evalCV = new Evaluation(train);
        evalCV.crossValidateModel(lr, train, 10, new Random(1));
        System.out.println("----- IBk 10-Fold Cross-Validation on Training Data -----");
        System.out.println(evalCV.toSummaryString());


    }
}
