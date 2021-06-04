package main.neighbors;

import main.Classifier;
import main.Dataset;
import main.util.Util;

import java.util.*;

public class KNN implements Classifier {
    private static final String NAME = "K-Nearest Neighbor";

    private final int k;
    private Dataset dataset;

    public KNN(int k) {
        this.k = k;
    }

    @Override
    public int classify(String[] example) {
        List<String[]> data = dataset.rows();
        Map<Double, Integer> distances = new TreeMap<>();
        for (int i = 0; i < data.size(); i++) {
            double distance = computeDistance(data.get(i), example);
            distances.put(distance, i);
        }

        int i = 0;
        List<Integer> votes = new ArrayList<>();
        for(Map.Entry<Double, Integer> entry : distances.entrySet()) {
            if(i >= k) {
                break;
            }

            int index = entry.getValue();
            int classValue = dataset.getClasses().indexOf(data.get(index)[dataset.getTarget()]);
            votes.add(classValue);

            i++;
        }

        return Util.getMostFrequentElement(votes);

    }

    @Override
    public void train(Dataset dataset) {
        this.dataset = dataset;
    }

    private double computeDistance(String[] e1, String[] e2) {
        double distance = 0;
        if (dataset.getType() == Dataset.DISCRETE) { // Hamming distance
            for (int i = 0; i < e1.length; i++) {
                if (i == dataset.getTarget()) continue;
                int error = e1[i].equals(e2[i]) ? 0 : 1;
                distance += error;
            }
        } else { // Euclidean distance
            double sum = 0.0;
            for (int i = 0; i < e1.length; i++) {
                if (i == dataset.getTarget()) continue;
                sum += Math.pow(Double.parseDouble(e1[i]) - Double.parseDouble(e2[i]), 2);
            }
            distance = Math.sqrt(sum);
        }
        return distance;
    }

    @Override
    public String toString() {
        return NAME + "(k:" + k + ")";
    }
}
