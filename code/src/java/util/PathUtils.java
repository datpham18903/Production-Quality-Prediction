package util;

import java.io.File;

public class PathUtils {
    public static final String DEFAULT_TRAIN_PATH = "src/datasets/train_data.arff";
    public static final String DEFAULT_TEST_PATH = "src/datasets/test_data.arff";
    public static final String CODE_DIR = "code";

    public static String[] resolveDataPaths(String[] args) {
        String trainPath = DEFAULT_TRAIN_PATH;
        String testPath = DEFAULT_TEST_PATH;
        
        if (args.length >= 1) {
            trainPath = args[0];
        }
        if (args.length >= 2) {
            testPath = args[1];
        }
        
        trainPath = resolvePath(trainPath);
        testPath = resolvePath(testPath);
        
        return new String[] { trainPath, testPath };
    }

    public static String resolvePath(String path) {
        File file = new File(path);
        if (file.isAbsolute()) {
            return path;
        }
        
        String currentDirPath = new File(System.getProperty("user.dir"), path).getAbsolutePath();
        if (fileExists(currentDirPath)) {
            return currentDirPath;
        }
        
        String workingDir = System.getProperty("user.dir");
        File codeDir = new File(workingDir, CODE_DIR);
        if (codeDir.exists() && codeDir.isDirectory()) {
            String codeDirPath = new File(codeDir, path).getAbsolutePath();
            if (fileExists(codeDirPath)) {
                return codeDirPath;
            }
        }
        
        File parentDir = new File(workingDir).getParentFile();
        if (parentDir != null) {
            String parentDirPath = new File(parentDir, path).getAbsolutePath();
            if (fileExists(parentDirPath)) {
                return parentDirPath;
            }
        }
        
        return currentDirPath;
    }

    public static boolean fileExists(String path) {
        File file = new File(path);
        return file.exists() && file.isFile();
    }

    public static String getParentDirectory(String filePath) {
        File file = new File(filePath);
        return file.getParent();
    }

    public static String combinePaths(String basePath, String relativePath) {
        File baseFile = new File(basePath);
        File combinedFile = new File(baseFile, relativePath);
        return combinedFile.getPath();
    }
}