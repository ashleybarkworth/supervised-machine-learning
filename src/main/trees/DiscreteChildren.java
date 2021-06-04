package main.trees;

import main.Dataset;
import main.math.Tuple;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class DiscreteChildren extends Children {

    private final List<Tuple<String, TreeNode>> nodes; // list of (attribute value, corresponding tree node)

    public DiscreteChildren(TreeNode parent, List<String> values, List<Integer> attributes) {
        super(parent, attributes);
        this.nodes = new ArrayList<>();
        for (String value : values) {
            this.nodes.add(new Tuple<>(value, new TreeNode(parent)));
        }
        this.parent = parent;
    }

    @Override
    public int size() {
        return nodes.size();
    }

    @Override
    public void split(List<Dataset> datasets) {
        for (int i = 0; i < nodes.size(); i++) {
            TreeNode node = nodes.get(i).second();
            // If there's no examples, then below this new branch add a leaf node
            // with label = most common target value in the examples
            if (datasets.get(i).isEmpty()) {
                node.setPrediction(this.parent.getPrediction());
                node.setLeaf(true);
            } else {
                node.split(datasets.get(i), getAttributes());
            }
        }
    }

    @Override
    public int predict(List<String> example) {
        int attribute = parent.getAttribute();
        for (Tuple<String, TreeNode> entry : nodes) {
            String value = entry.first();
            TreeNode child = entry.second();
            if (example.get(attribute).equals(value)) {
                return child.classify(example);
            }
        }

        return parent.getPrediction();
    }

    @Override
    public TreeNode get(int i) {
        return nodes.get(i).second();
    }

    @Override
    public Iterator<TreeNode> iterator() {
        return new Iterator<TreeNode>() {
            private Integer index = 0;

            @Override
            public boolean hasNext() {
                return index < nodes.size();
            }

            @Override
            public TreeNode next() {
                return get(index++);
            }
        };
    }
}
