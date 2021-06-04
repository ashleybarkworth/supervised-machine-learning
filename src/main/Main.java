package main;

import main.bayes.NaiveBayes;
import main.ensemble.AdaBoost;
import main.ensemble.RandomForest;
import main.math.Statistics;
import main.neighbors.KNN;
import main.trees.ID3;
import main.util.DataUtil;

import java.util.*;

public class Main {

    public static final String carData = "car.data";
    public static final String mushroomData = "mushroom.data";
    public static final String letterRecognitionData = "letter-recognition.data";
    public static final String ecoliData = "ecoli.data";
    public static final String breastCancerData = "breast-cancer-wisconsin.data";

    private static final int NUMBER_OF_K_FOLD_ITERATIONS = 10;
    private static final int K_FOLD = 5;
    private static final KFold cv = new KFold(K_FOLD);


    public static void main(String[] args) {
        try {
            Map<Dataset, List<Classifier>> datasetAndAlgorithms = new HashMap<>();

            // Car data
            Dataset carDataset = DataUtil.parseFile(carData);
            carDataset.setName("Car Dataset");
            List<Classifier> classifiers = new ArrayList<>();
            classifiers.add(new ID3());
            classifiers.add(new AdaBoost(new ID3(1), 200, 0.8));
            classifiers.add(new RandomForest(new ID3(Integer.MAX_VALUE,4,1), 0.6, 200));
            classifiers.add(new NaiveBayes());
            classifiers.add(new KNN(2));
           datasetAndAlgorithms.put(carDataset, classifiers);

            // Mushroom data
            Dataset mushroomDataset = DataUtil.parseFile(mushroomData);
            mushroomDataset.setName("Mushroom Dataset");
            classifiers = new ArrayList<>();
            classifiers.add(new ID3());
            classifiers.add(new AdaBoost(new ID3(1),50,0.5));
            classifiers.add(new RandomForest(new ID3(Integer.MAX_VALUE, 4, 1), 0.7, 100));
            classifiers.add(new NaiveBayes());
            classifiers.add(new KNN(1));
           datasetAndAlgorithms.put(mushroomDataset, classifiers);

            // Letter recognition data
            Dataset letterRecognitionDataset = DataUtil.parseFile(letterRecognitionData);
            letterRecognitionDataset.setName("Letter Recognition Dataset");
            classifiers = new ArrayList<>();
            classifiers.add(new ID3());
            classifiers.add(new AdaBoost(new ID3(1),100,0.8));    // Warning: this takes over 15 minutes for 10 iterations
            classifiers.add(new RandomForest(new ID3(8, 6, 2), 0.5, 100)); // Warning: this takes over 10 minutes for 10 iterations
            classifiers.add(new NaiveBayes());
            classifiers.add(new KNN(1));
            datasetAndAlgorithms.put(letterRecognitionDataset, classifiers);

            // Ecoli data
            Dataset ecoliDataset = DataUtil.parseFile(ecoliData);
            ecoliDataset.setName("Ecoli Dataset");
            classifiers = new ArrayList<>();
            classifiers.add(new ID3());
            classifiers.add(new AdaBoost(new ID3(1), 200,0.6));
            classifiers.add(new RandomForest(new ID3(Integer.MAX_VALUE, 4, 1), 0.6, 200));
            classifiers.add(new NaiveBayes());
            classifiers.add(new KNN(10));
           datasetAndAlgorithms.put(ecoliDataset, classifiers);

           // Breast cancer data
            Dataset breastCancerDataset = DataUtil.parseFile(breastCancerData);
            breastCancerDataset.setName("Breast Cancer Dataset");
            classifiers = new ArrayList<>();
            classifiers.add(new ID3());
            classifiers.add(new AdaBoost(new ID3(1),200,0.6));
            classifiers.add(new RandomForest(new ID3(Integer.MAX_VALUE, 4, 3), 0.6, 150));
            classifiers.add(new NaiveBayes());
            classifiers.add(new KNN(1));
            datasetAndAlgorithms.put(breastCancerDataset, classifiers);

            // For each data set, run the five learning algorithms and print the results
            for (Map.Entry<Dataset, List<Classifier>> entry : datasetAndAlgorithms.entrySet()) {
                Dataset dataset = entry.getKey();
                System.out.println(dataset);
                System.out.println("---------------------------------------");
                List<Classifier> algorithms = entry.getValue();
                for (Classifier algorithm : algorithms) {
                    System.out.println(algorithm);
                    crossValidate(dataset, algorithm);
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void crossValidate(Dataset dataset, Classifier algorithm) {
        List<Double> allAccuracies = new ArrayList<>();
        for (int i=1; i<=NUMBER_OF_K_FOLD_ITERATIONS; i++) {
            System.out.print(i);
            cv.init(dataset, algorithm);
            List<Double> accuracies = cv.crossValidate();
            allAccuracies.addAll(accuracies);
        }
        System.out.println();
        printStatistics(allAccuracies);
    }

    private static void printStatistics(List<Double> accuracies) {
        double average = 100 * Statistics.computeAverage(accuracies);
        double standardDeviation = 100 * Statistics.computeStandardDeviation(accuracies);
        System.out.println("Average: " + average);
        System.out.println("Standard deviation: " + standardDeviation);
        System.out.println();
    }

}
