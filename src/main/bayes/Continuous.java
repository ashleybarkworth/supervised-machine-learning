package main.bayes;

import java.util.List;

import static java.lang.Math.*;

public class Continuous implements Attribute {

    private final double mean;
    private final double standardDeviation;

    public Continuous(List<String> values) {
        this.mean = computeMean(values);
        this.standardDeviation = computeStandardDeviation(values);
    }

    @Override
    public double probability(String value) {
        if (pow(standardDeviation, 2) == 0) {
            return 0;
        }
        double val = Double.parseDouble(value);
        double exponent = java.lang.Math.exp(-pow(val-mean, 2)/(2*pow(standardDeviation, 2)));
        return (1 / (sqrt(2*PI) * standardDeviation)) * exponent;
    }

    private double computeMean(List<String> values) {
        double sum = 0;
        for (String val : values) {
            sum += Double.parseDouble(val);
        }
        return sum/values.size();
    }

    private double computeStandardDeviation(List<String> values) {
        double variance = 0.0;
        for (String value : values) {
            double val = Double.parseDouble(value);
            variance += Math.pow(val - mean, 2);
        }
        variance /= values.size();
        return Math.sqrt(variance);
    }

}
