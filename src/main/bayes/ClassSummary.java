package main.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ClassSummary {

    private final int classValue;               // the int value of the class (i.e. the index in classes() from dataset)
    private final double classProbability;      // the probability of the class value
    private final List<Attribute> attributes;   // list of Attribute instances

    public ClassSummary(int classValue, double classProbability, List<Attribute> attributes) {
        this.classValue = classValue;
        this.classProbability = classProbability;
        this.attributes = attributes;
    }

    public int getClassValue() {
        return classValue;
    }

    public double probability(String[] e, int target) {
        List<Double> conditionalProbabilities = new ArrayList<>(Arrays.asList(new Double[attributes.size()]));
        List<String> attrs = new ArrayList<>(Arrays.asList(e));
        attrs.remove(target);
        for (int i = 0; i < attributes.size(); i++) {
            String attributeValue = attrs.get(i);
            double probability = attributes.get(i).probability(attributeValue);
            conditionalProbabilities.set(i, probability);
        }
        double product = 1.0;
        for (Double conditionalProbability : conditionalProbabilities) {
            product *= conditionalProbability;
        }
        return product * classProbability;
    }


}
