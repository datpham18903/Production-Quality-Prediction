/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.weka;
import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.Evaluation;

/**
 *
 * @author DB
 */
public class ZeroRf10 {
    public static void main(String[] args) throws Exception {

        Instances train = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\data_compressed.arff");


        train.setClassIndex(train.numAttributes() - 1);

        ZeroR zero = new ZeroR();
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(zero, train, 10, new Random(1));

        
        System.out.println("----- ZeroR 10-Fold Cross-Validation Results -----");
        System.out.println(eval.toSummaryString());


    }
}
