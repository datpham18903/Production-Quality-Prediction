@echo off
echo Copying Maven dependencies...
call mvn dependency:copy-dependencies

echo Compiling SampleTesting...
REM Include all the Maven dependencies in the classpath during compilation
javac -cp ".;target/classes;target/dependency/*" src/java/others/SampleTesting.java -d target/classes

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
)

echo Running SampleTesting...

REM Check if model name is provided
if "%~1"=="" (
    echo No model specified. Using default LINEARREGRESSION.
) else (
    echo Testing with model: %1
)

REM Run with proper classpath including all Maven dependencies and pass arguments
java -cp ".;target/classes;target/dependency/*" others.SampleTesting %*

echo Done. 