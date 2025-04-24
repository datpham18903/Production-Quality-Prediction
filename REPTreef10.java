/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.weka;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;


/**
 *
 * @author DB
 */
public class REPTreef10 {
    public static void main(String[] args) throws Exception {

        Instances filteredTrain = ConverterUtils.DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\data_compressed.arff");

        filteredTrain.setClassIndex(filteredTrain.numAttributes() - 1);

        REPTree rep = new REPTree();
        
        Evaluation eval = new Evaluation(filteredTrain);
        eval.crossValidateModel(rep, filteredTrain, 10, new Random(1));

        System.out.println("----- REPTree fold 10 Results -----");
        System.out.println(eval.toSummaryString());
        
    }
}
