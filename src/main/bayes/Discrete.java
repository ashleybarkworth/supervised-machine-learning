package main.bayes;

import main.util.Util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Discrete implements Attribute {

    private final double unseenConditionalProbability;          // the unseen conditional probability
    private final Map<String, Double> conditionalProbabilities; // the conditional probability of an attribute value

    public Discrete(List<String> values, int attributeCount) {
        int valuesAttributeCount = values.size() + attributeCount;
        // Compute the unseen conditional probability for attribute values that aren't encountered in training
        unseenConditionalProbability = ((double)1)/(valuesAttributeCount);
        // Get the frequency of each attribute value in the examples
        Map<String, Integer> frequencyMap = Util.getFrequencyMap(values);
        // Calculate the conditional probability for each attribute
        conditionalProbabilities = new HashMap<>();
        for (Map.Entry<String, Integer> entry: frequencyMap.entrySet()) {
            double conditionalProbability = (((double)entry.getValue()+1)) / (valuesAttributeCount);
            conditionalProbabilities.put(entry.getKey(), conditionalProbability);
        }
    }

    @Override
    public double probability(String value) {
        Double probability = conditionalProbabilities.get(value);
        if (probability == null) {
            conditionalProbabilities.put(value, unseenConditionalProbability);
            probability = conditionalProbabilities.get(value);
        }
        return probability;
    }
}
