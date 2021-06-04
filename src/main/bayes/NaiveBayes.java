package main.bayes;

import main.Classifier;
import main.Dataset;

import java.util.*;

import static main.Dataset.CONTINUOUS;

public class NaiveBayes implements Classifier {
    private static final String NAME = "Naïve Bayes";

    private List<ClassSummary> summaries;
    private int target;

    public NaiveBayes() {}

    @Override
    public int classify(String[] example) {
        ClassSummary maxSummary = null;
        double maxProbability = -1;
        for (ClassSummary summary : summaries) {
            double probability = summary.probability(example, target);
            if (maxSummary == null || probability > maxProbability) {
                maxSummary = summary;
                maxProbability = probability;
            }
        }
        assert maxSummary != null;
        return maxSummary.getClassValue();
    }

    @Override
    public void train(Dataset dataset) {
        summaries = new ArrayList<>();
        target = dataset.getTarget();
        Map<Integer, Dataset> separated = dataset.splitByClass();

        for(Map.Entry<Integer, Dataset> entry : separated.entrySet()) {
            List<Attribute> attributes = new ArrayList<>();

            int classValue = entry.getKey();
            Dataset d = entry.getValue();
            double classProbability = ((double) d.getNumberOfExamples()) / dataset.getNumberOfExamples();

            for(int j = 0; j < d.getNumAttributes(); j++) {
                int numAttributeValues = d.getUniqueAttributeValues()[j].length;
                if (j == target) continue;
                List<String> attributeValues = d.getAttributeColumn(j);
                if (d.getType() == CONTINUOUS) {
                    attributes.add(new Continuous(attributeValues));
                } else {
                    attributes.add(new Discrete(attributeValues, numAttributeValues));
                }
            }
            summaries.add(new ClassSummary(classValue, classProbability, attributes));
        }
    }

    @Override
    public String toString() {
        return NAME;
    }

}
