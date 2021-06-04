package main.math;
import java.util.List;

public class Statistics {

    public static double computeAccuracy(List<String[]> examples, String[] predictions, int target) {
        int correct = 0;
        for (int i=0; i<predictions.length; i++) {
            if (predictions[i].equals(examples.get(i)[target])) {
                correct++;
            }
        }
        return ((double) correct) / predictions.length;
    }

    public static double computeAverage(List<Double> values) {
        double mean = 0.0;
        for (double value : values) {
            mean += value;
        }
        mean /= values.size();
        return mean;
    }

    public static double computeStandardDeviation(List<Double> values) {
        double mean = computeAverage(values);
        double variance = 0.0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= (values.size()-1);

        return Math.sqrt(variance);
    }
}
