@echo off
setlocal enabledelayedexpansion

REM Change to the code directory where pom.xml is located
cd "%~dp0"

if "%1"=="" goto usage

REM Define the package and class based on the algorithm
set "mainClass="

if /i "%1"=="IBk" (
    set "mainClass=baseline.IBkModel"
) else if /i "%1"=="LinearRegression" (
    set "mainClass=baseline.LinearRegressionModel"
) else if /i "%1"=="M5P" (
    set "mainClass=baseline.M5PModel"
) else if /i "%1"=="REPTree" (
    set "mainClass=baseline.REPTreeModel"
) else if /i "%1"=="ZeroR" (
    set "mainClass=baseline.ZeroRModel"
) else if /i "%1"=="RandomForest" (
    set "mainClass=clustering.RandomForestModel"
) else if /i "%1"=="SMOreg" (
    set "mainClass=clustering.SMOregModel"
) else if /i "%1"=="Bagging" (
    set "mainClass=ensemble.BaggingModel"
) else if /i "%1"=="Stacking" (
    set "mainClass=ensemble.StackingModel"
) else if /i "%1"=="MLP" (
    set "mainClass=clustering.MultiLayerPerceptronModel"
) else if /i "%1"=="SimpleKMeans" (
    set "mainClass=clustering.SimpleKMeansModel"
) else (
    echo Unknown algorithm: %1
    goto usage
)

REM Execute Maven with the appropriate class
call mvn exec:java -Dexec.mainClass=!mainClass!
exit /b 0

:usage
echo Usage: run.bat [algorithm]
echo Available algorithms:
echo   IBk - k-nearest neighbor classifier
echo   LinearRegression - linear regression model
echo   M5P - M5P decision tree for regression
echo   REPTree - fast decision tree learner
echo   ZeroR - predicts the mean value (baseline)
echo   RandomForest - random forest classifier
echo   SMOreg - support vector machine for regression
echo   Bagging - bagging ensemble method
echo   Stacking - stacking ensemble method
echo   MLP - multilayer perceptron neural network
echo   SimpleKMeans - k-means clustering
exit /b 1 