package main.trees;

import main.Dataset;

import java.util.List;

public abstract class Children implements Iterable<TreeNode> {
    TreeNode parent;
    private final List<Integer> attributes;

    public Children(TreeNode parent, List<Integer> attributes) {
        this.parent = parent;
        this.attributes = attributes;
    }

    abstract public int size();
    abstract public void split(List<Dataset> datasets);
    abstract public int predict(List<String> example);
    abstract public TreeNode get(int i);
    public List<Integer> getAttributes() { return attributes; }
}
