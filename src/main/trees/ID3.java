package main.trees;

import main.Classifier;
import main.Dataset;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ID3 implements Classifier {
    private static final String NAME = "ID3";
    private Dataset pruningDataset;
    private TreeNode root;
    int maxLevel;
    int maxFeatures;
    int minNumberExamples;

    public ID3() {
        initialize(Integer.MAX_VALUE, -1, 1);
    }

    public ID3(int maxLevel) {
        initialize(maxLevel, -1, 1);
    }

    public ID3(int maxLevel, int minNumberExamples) {
        initialize(maxLevel, -1, minNumberExamples);
    }

    public ID3(int maxLevel, int maxFeatures, int minNumberExamples) {
        initialize(maxLevel, maxFeatures, minNumberExamples);
    }

    public ID3 createTree() {
        return new ID3(this.maxLevel, -1, this.minNumberExamples);
    }

    public ID3 createRandomTree() {
        return new ID3(this.maxLevel, this.maxFeatures, this.minNumberExamples);
    }

    private void initialize(int maxLevel, int m, int minNumberExamples) {
        root = null;
        this.maxLevel = maxLevel;
        this.maxFeatures = m;
        this.minNumberExamples = minNumberExamples;
    }

    public void setPruningDataset(Dataset pruningDataset) { this.pruningDataset = pruningDataset; }

    public int classify(String[] example) {
        return root.classify(new ArrayList<>(Arrays.asList(example)));
    }

    public void train(Dataset dataset) {
        // Create new root for the decision tree
        if (maxFeatures > 0) {
            root = new TreeNode(maxLevel, maxFeatures, minNumberExamples);
        } else {
            root = new TreeNode(maxLevel);
        }
        // The indices of the attributes (minus the target attribute)
        List<Integer> attributes = IntStream.range(0, dataset.getNumAttributes()).boxed().collect(Collectors.toList());
        attributes.remove(Integer.valueOf(dataset.getTarget()));
        // Split root node
        root.split(dataset, attributes);
        // Apply reduced error pruning for ID3 only (not AdaBoost or Random Forest)
        if (pruningDataset != null) {
            Pruning pruning = new Pruning(pruningDataset, root);
            pruning.pruneTree(root);
        }
    }


    @Override
    public String toString() {
        return NAME;
    }

}