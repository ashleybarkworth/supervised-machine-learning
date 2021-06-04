package main.ensemble;

import main.Classifier;
import main.Dataset;
import main.math.Tuple;
import main.trees.ID3;
import main.util.Util;

import java.util.*;

import static java.lang.Math.exp;
import static java.lang.Math.log;

public class AdaBoost implements Classifier {
    private static final String NAME = "AdaBoost";

    private final ID3 algorithm;
    private List<String> classes;
    private List<Double> alphas;
    private List<ID3> decisionTrees;
    private final double proportionOfSamples;
    private final int numEstimators;
    private static final Double EPSILON = 0.001;

    public AdaBoost(ID3 algorithm, int numEstimators, double proportionOfSamples) {
        this.algorithm = algorithm;
        this.numEstimators = numEstimators;
        this.proportionOfSamples = proportionOfSamples;
    }

    @Override
    public void train(Dataset dataset) {
        classes = dataset.getClasses();
        int classCount = classes.size();
        List<String[]> data = dataset.rows();
        int totalNumberExamples = data.size();
        int numberOfSamples = (int)(dataset.getNumberOfExamples() * proportionOfSamples);
        List<Double> weights = new ArrayList<>(Arrays.asList(new Double[totalNumberExamples]));
        Collections.fill(weights, 1.0/totalNumberExamples);
        int target = dataset.getTarget();

        for (int i = 0; i < numEstimators; i++) {
            // Sample with replacement to create new training set
            List<Integer> indices = Util.generateIndices(weights, numberOfSamples);
            Dataset weightedDataset = new Dataset(dataset, indices);
            ID3 id3 = algorithm.createTree();
            id3.train(weightedDataset);

            // Predict using weak learner and calculate weight sum as well as error sum using predictions
            double errorSum = 0.0;
            double weightSum = 0.0;
            String[] predictions = new String[weights.size()];
            for (int k = 0; k < totalNumberExamples; k++) {
                String prediction = classes.get(id3.classify(data.get(k)));
                String actual = data.get(k)[target];
                if (!actual.equals(prediction)) {
                    errorSum += weights.get(k);
                }
                weightSum += weights.get(k);
                predictions[k] = prediction;
            }

            // Compute error
            double error = errorSum/weightSum;

            // Compute alpha
            double alpha = log((1.0-error)/(error)) + log(classCount - 1);

            if(Double.isInfinite(alpha)) {
                alpha = log((1.0-error+EPSILON)/(error+EPSILON)) + log(classCount - 1);
            }

            // Update weights
            weightSum = 0.0;
            for (int k = 0; k < weights.size(); k++) {
                double weight = weights.get(k);
                if (!data.get(k)[target].equals(predictions[k])) {
                    weights.set(k, weight * exp(alpha));
                }
                weightSum += weights.get(k);

            }

            // Normalize weights
            for (int k = 0; k < weights.size(); k++) {
                weights.set(k, weights.get(k)/weightSum);
            }

            // Add classifier and alpha to lists
            add(id3, alpha);
        }
    }

    @Override
    public int classify(String[] example) {
            Map<Integer, Double> probabilities = new LinkedHashMap<>();
            // Take a weighted "vote" for each possible class
            for (int i = 0; i < classes.size(); i++) {
                for (int j = 0; j < decisionTrees.size(); j++) {
                    Integer classification = decisionTrees.get(j).classify(example);
                    Double probability = probabilities.computeIfAbsent(classification, k -> 0.0);
                    probability += alphas.get(j);
                    probabilities.replace(classification, probability);
                }
            }
            // Return the class with the largest "vote"
            Tuple<Integer, Double> max = null;
            for (Map.Entry<Integer, Double> probability : probabilities.entrySet()) {
                if (max == null || max.second() < probability.getValue()) {
                    max = new Tuple<>(probability.getKey(), probability.getValue());
                }
            }

            return max.first();
    }

    public void add(ID3 c, double alpha) {
        if(decisionTrees == null) {
            decisionTrees = new ArrayList<>();
            alphas = new ArrayList<>();
        }
        decisionTrees.add(c);
        alphas.add(alpha);
    }

    @Override
    public String toString() {
        return NAME + "(estimators:"+numEstimators+", proportion:"+proportionOfSamples+")";
    }

}
