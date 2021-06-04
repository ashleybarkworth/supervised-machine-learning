package main.ensemble;

import main.Classifier;
import main.Dataset;
import main.trees.ID3;
import main.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class RandomForest implements Classifier {
    private static final String NAME = "Random Forest";
    private final ID3 algorithm;
    private final int numTrees;
    private final double proportionOfSamples;
    private List<ID3> trees;

    public RandomForest(ID3 algorithm, double proportionOfSamples, int numTrees) {
        this.algorithm = algorithm;
        this.proportionOfSamples = proportionOfSamples;
        this.numTrees = numTrees;
    }

    public ID3 getAlgorithm() { return algorithm; }

    @Override
    public int classify(String[] example) {
        List<Integer> predictions = new ArrayList<>(Arrays.asList(new Integer[trees.size()]));
        for(int i = 0; i < trees.size(); i++) {
            predictions.set(i, trees.get(i).classify(example));
        }

        return Util.getMostFrequentElement(predictions);
    }

    @Override
    public void train(Dataset dataset) {
        int numberOfSamples = (int) (dataset.getNumberOfExamples() * proportionOfSamples);
        List<Double> weights = new ArrayList<>(Arrays.asList(new Double[dataset.getNumberOfExamples()]));
        Collections.fill(weights, 1./dataset.getNumberOfExamples());

        for (int i = 0; i < numTrees; i++) {
            List<Integer> indices = Util.generateIndices(weights, numberOfSamples);
            Dataset weightedDataset = new Dataset(dataset, indices);
            ID3 id3 = algorithm.createRandomTree();
            id3.train(weightedDataset);
            this.add(id3);
        }
    }

    public void add(ID3 tree) {
        if(trees == null) {
            trees = new ArrayList<>();
        }
        trees.add(tree);
    }

    @Override
    public String toString() {
        return NAME + "(size:"+numTrees+", percentage:"+proportionOfSamples+")";
    }
}
