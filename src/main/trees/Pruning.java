package main.trees;

import main.Dataset;
import main.math.Statistics;

import java.util.Arrays;
import java.util.List;

public class Pruning {

    private final TreeNode root;                    // the root of the decision tree
    private final List<String> classes;             // the list of unique class values
    private final List<String[]> pruningExamples;   // the list of pruning examples
    private double maxAccuracy;                     // the maximum accuracy obtained from classifying pruning exmaples
    private final int target;                       // the target attribute column

    public Pruning(Dataset testDataset, TreeNode root) {
        this.root = root;
        this.classes = testDataset.getClasses();
        pruningExamples = testDataset.rows();
        target = testDataset.getTarget();
        maxAccuracy = getAccuracy();
    }

    /**
     * Recursively prunes tree by removing each node one by one and checking if the resulting accuracy
     * is greater than the accuracy of the original decision tree. If it is, then the remove is permanent.
     * If not, pruneTree() is called on each of the node's children
     *
     * @param treeNode      a node in the decision tree
     */
    public void pruneTree(TreeNode treeNode) {
        Children children = treeNode.children;
        if (children == null) return;

        treeNode.setLeaf(true);
        double newAccuracy = getAccuracy();

        if (newAccuracy > maxAccuracy) {
            maxAccuracy = newAccuracy;
            return;
        }

        treeNode.setLeaf(false);
        for (TreeNode child : children) {
            pruneTree(child);
        }

    }

    /**
     * Calculates the accuracy of the root's classifications on the list of pruning examples
     * @return  accuracy value between 0 and 1
     */
    public double getAccuracy() {
        String[] predictions = new String[pruningExamples.size()];
        for (int i = 0; i < pruningExamples.size(); i++) {
            String[] example = pruningExamples.get(i);
            int prediction = root.classify(Arrays.asList(example));
            predictions[i] = classes.get(prediction);
        }
        return Statistics.computeAccuracy(pruningExamples, predictions, target);
    }

}
