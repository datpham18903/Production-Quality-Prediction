 /*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.weka;
import weka.classifiers.Evaluation;
import weka.core.Instances; 
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.core.SelectedTag;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;
/**
 *  
 * @author DB
 */
public class LinearRegressionRun {

    public static void main(String[] args) throws Exception {
       
        Instances train = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\merged_train.arff");
        Instances test = DataSource.read("C:\\Users\\DB\\Documents\\NetBeansProjects\\weka\\src\\main\\java\\data\\merged_test.arff");

        // Set target attribute (last column)
        

        Remove remove = new Remove();
        remove.setAttributeIndices("1"); // Adjust indices if needed
        //remove.setAttributeIndices("19-22");
        remove.setInputFormat(train);
        Instances filteredTrain = Filter.useFilter(train, remove);
        Instances filteredTest = Filter.useFilter(test, remove);
        
        filteredTrain.setClassIndex(filteredTrain.numAttributes() - 1);
        filteredTest.setClassIndex(filteredTest.numAttributes() - 1);
        
        // Run Linear Regression
        LinearRegression lr = new LinearRegression();
        lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));

        lr.buildClassifier(filteredTrain);
        Evaluation evalLR = new Evaluation(filteredTrain);
        evalLR.evaluateModel(lr, filteredTest);
        
        System.out.println("----- Linear Regression Results -----");
        System.out.println(evalLR.toSummaryString());
    }
    }

