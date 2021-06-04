package main.trees;

import main.Dataset;
import main.math.Tuple;
import main.util.Util;

import java.util.*;

import static main.Dataset.CONTINUOUS;
import static main.Dataset.DISCRETE;

public class TreeNode {
    private int attribute;          // index of attribute value
    private boolean leaf;           // leaf or internal node?
    private int prediction;         // class prediction
    private int maxLevel;           // maximum level for splitting
    private int level;              // current level of tree
    private int maxFeatures;        // number of attributes to select from for Random Forest
    private int minNumberExamples;  // minimum number of examples in each node
    Children children;              // children nodes

    public TreeNode(int maxLevel) {
        this.initialize(0, maxLevel,-1, 1);
    }

    public TreeNode(TreeNode parent) {
        this.initialize(parent.level+1, parent.maxLevel, parent.maxFeatures, parent.minNumberExamples);
    }

    public TreeNode(int maxLevel, int m, int minNumberExamples) {
        this.initialize(0, maxLevel, m, minNumberExamples);
    }

    private void initialize(int level, int maxLevel, int m, int minNumberExamples) {
        leaf = false;
        this.attribute = -1;
        this.prediction = -1;
        this.maxLevel = maxLevel;
        this.level = level;
        this.maxFeatures = m;
        this.minNumberExamples = minNumberExamples;
    }

    public int getAttribute() {
        return attribute;
    }

    public void setLeaf(boolean leaf) { this.leaf = leaf; }

    public int getPrediction() { return prediction; }

    public void setPrediction(int prediction) {
        this.prediction = prediction;
    }

    public void split(Dataset dataset, List<Integer> attributes) {
        double initialEntropy = dataset.entropy();
        List<String> classes = dataset.getClasses();
        // Set this node's prediction to the most common class (i.e. target attribute value) in the examples
        this.prediction = classes.indexOf(Util.getMostFrequentElement(dataset.getAttributeColumn(dataset.getTarget())));

        // If we are the maximum depth, the dataset is perfectly classified, or there aren't enough examples then make node a leaf
        if (level == maxLevel || initialEntropy == 0 || dataset.getNumberOfExamples() <= Integer.max(1, minNumberExamples)) {
            leaf = true;
            return;
        }

        double minEntropy = -1;
        int bestAttribute = -1;

        // If ID3 is being performed on a Random Forest tree node, select a set of random attributes to split with
        List<Integer> attributeIndices;
        if (maxFeatures > 0) {
            attributeIndices = pickRandomFeatures(attributes, maxFeatures);
        // Otherwise consider all attributes for splitting
        } else {
            attributeIndices = attributes;
        }

        // Loop over each candidate attribute and determine the best attribute based on minimum entropy in the
        // resulting datasets (this is equivalent to calculating the maximum information gain for the current dataset)
        for (int attribute : attributeIndices) {
            double entropy = 0.0;
            if (dataset.getType() == DISCRETE) {
                Tuple<List<String>, List<Dataset>> subsets = dataset.splitByDiscreteAttribute(attribute);
                entropy = Dataset.calculateWeightedEntropy(subsets.second());
            } else if (dataset.getType() == CONTINUOUS) {
                Tuple<Double, Tuple<Dataset, Dataset>> subsets = dataset.splitByContinuousAttribute(attribute);
                if (subsets == null) continue;
                entropy = Dataset.calculateWeightedEntropy(subsets.second());
            } else {
                System.err.println("Unknown data type");
                System.exit(-1);
            }

            if(bestAttribute < 0 || entropy < minEntropy) {
                minEntropy = entropy;
                bestAttribute = attribute;
            }
        }

        if (bestAttribute < 0) {
            this.leaf = true;
            return;
        }

        if (initialEntropy <= minEntropy) {
            this.leaf = true;
            return;
        }

        attribute = bestAttribute;
        List<Integer> newAttributes = new ArrayList<>(attributes);
        newAttributes.remove(Integer.valueOf(bestAttribute));

        // Create new children nodes, using the best attribute to split the training examples
        List<Dataset> subsets = new ArrayList<>();
        if (dataset.getType() == CONTINUOUS) {
            Tuple<Double, Tuple<Dataset, Dataset>> split = dataset.splitByContinuousAttribute(bestAttribute);
            Dataset under = split.second().first();
            Dataset over = split.second().second();
            subsets.add(under);
            subsets.add(over);
            children = new ContinuousChildren(this, split.first(), newAttributes);
        } else if (dataset.getType() == DISCRETE) {
            Tuple<List<String>, List<Dataset>> split = dataset.splitByDiscreteAttribute(bestAttribute);
            subsets = split.second();
            children = new DiscreteChildren(this, split.first(), newAttributes);
        }
        // Split the children nodes
        children.split(subsets);
    }

    public int classify(List<String> example) {
        if (this.leaf) {
            return this.getPrediction();
        } else {
            return children.predict(example);
        }
    }

    /**
     * Used for Random Forest to randomly choose m attributes
     * @param features      the list of attributes to choose from
     * @param m             the number of attributes to choose
     * @return              m random attributes
     */
    public List<Integer> pickRandomFeatures(List<Integer> features, int m) {
        List<Integer> copy = new ArrayList<>(features);
        Collections.shuffle(copy);
        return m > copy.size() ? copy.subList(0, copy.size()) : copy.subList(0, m);
    }

}
