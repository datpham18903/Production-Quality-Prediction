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
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author DB
 */
public class REPTreef10 {
    public static void main(String[] args) throws Exception {

        Instances train = ConverterUtils.DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\merged_train.arff");

        train.setClassIndex(train.numAttributes() - 1);

        Remove remove = new Remove();
        remove.setAttributeIndices("1");
        remove.setInputFormat(train);
        Instances filteredTrain = Filter.useFilter(train, remove);

        REPTree rep = new REPTree();
        
        Evaluation eval = new Evaluation(filteredTrain);
        eval.crossValidateModel(rep, filteredTrain, 10, new Random(1));

        System.out.println("----- REPTree fold 10 Results -----");
        System.out.println(eval.toSummaryString());
        
    }
}
