/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.weka;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.M5P;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.SelectedTag;

import java.util.Random;

/**
 *
 * @author DB
 */
public class M5Pf10 {
    public static void main(String[] args) throws Exception {

        Instances train = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\data_compressed.arff");



        train.setClassIndex(train.numAttributes() - 1);

        M5P m5p = new M5P();
        
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(m5p, train, 10, new Random(1));

        
        System.out.println("----- M5P 10-Fold Cross-Validation Results -----");
        System.out.println(eval.toSummaryString());


    }
}
