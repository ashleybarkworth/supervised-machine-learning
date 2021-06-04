package main.trees;

import main.Dataset;

import java.util.Iterator;
import java.util.List;

public class ContinuousChildren extends Children {
    private final double pivot;       // Threshold to split examples
    private final TreeNode under;     // Contains examples less than the threshold
    private final TreeNode over;      // Contains examples greater than the threshold

    public ContinuousChildren(TreeNode parent, double pivot, List<Integer> attributes) {
        super(parent, attributes);
        this.pivot = pivot;
        under = new TreeNode(parent);
        over = new TreeNode(parent);
    }

    @Override
    public int size() {
        return 2;
    }

    @Override
    public void split(List<Dataset> datasets) {
        under.split(datasets.get(0), getAttributes());
        over.split(datasets.get(1), getAttributes());
    }

    @Override
    public int predict(List<String> example) {
        int attribute = parent.getAttribute();
        if (Double.parseDouble(example.get(attribute)) < pivot) {
            return under.classify(example);
        } else {
            return over.classify(example);
        }
    }

    @Override
    public TreeNode get(int i) {
        return i == 0 ? under : over;
    }

    @Override
    public Iterator<TreeNode> iterator() {
        return new Iterator<TreeNode>() {
            private Integer index = 0;

            @Override
            public boolean hasNext() {
                return index < 2;
            }

            @Override
            public TreeNode next() {
                return index++ == 0 ? under : over;
            }
        };
    }
}
