package main.util;

import java.util.*;

import static java.lang.Math.random;

public class Util {

    /**
     * Returns a HashMap that maps an element in the given list to the number of the times that it
     * occurs in the list, e.g. {"red","blue","red","yellow","blue","blue"} would return
     * "red" -> 2, "yellow" -> 1, "blue" -> 3
     *
     * @param list      the list of elements
     * @return          a frequency map of the elements in the list
     */
    public static Map<String, Integer> getFrequencyMap(List<String> list) {
        Map<String, Integer> map = new HashMap<>();

        for (String element : list) {
            Integer count = map.get(element);
            if (count == null) {
                count = 1;
                map.put(element, count);
            } else {
                count++;
                map.replace(element, count);
            }
        }

        return map;
    }

    /**
     * Returns the most frequently occurring element in the given list
     * @param list      the list of elements
     * @param <T>       the type of element in the list (e.g. String, Integer)
     * @return          the most frequent element in the list
     */
    public static <T> T getMostFrequentElement(List<T> list) {
        Map<T, Integer> map = new LinkedHashMap<>();

        for (T t : list) {
            Integer val = map.get(t);
            if (val == null) {
                map.put(t, 1);
            } else {
                map.replace(t, val + 1);
            }
        }

        Map.Entry<T, Integer> max = null;

        for (Map.Entry<T, Integer> e : map.entrySet()) {
            if (max == null|| e.getValue() > max.getValue())
                max = e;
        }
        return max.getKey();
    }

    /**
     * Used for bootstrapping (i.e. sampling with replacement) by generating indices
     * from the original dataset between 0 and numberOfIndices that are determined in
     * part by randomness (i.e. random()) and in part by the weights assigned to each index.
     *
     * @param weights           the weights assigned to each index. Higher weights increase the
     *                          selection of those indices
     * @param numberOfIndices   the total number of indices to generate
     * @return                  the list of generated indices
     */
    public static List<Integer> generateIndices(List<Double> weights, int numberOfIndices) {
        List<Double> probabilities = Arrays.asList(new Double[weights.size()]);

        // Normalize weights
        double sum = 0.0;
        for (Double weight : weights) {
            sum += weight;
        }
        for (int i = 0; i < weights.size(); i++) {
            probabilities.set(i, weights.get(i)/sum);
        }

        // Sample indices with replacement based on cdf given by weights (i.e. probabilities)
        List<Integer> indices = Arrays.asList(new Integer[numberOfIndices]);
        for (int i = 0; i < numberOfIndices; i++) {
            double rand = random();
            double cumulativeProbability = 0.0;
            for (int j = 0; j < probabilities.size(); j++) {
                cumulativeProbability += probabilities.get(j);
                if (rand <= cumulativeProbability) {
                    indices.set(i, j);
                    break;
                }
            }
        }
        return indices;
    }
}
