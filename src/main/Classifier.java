package main;

public interface Classifier {
    int classify(String[] example);
    void train(Dataset dataset);
}
