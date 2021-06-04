package main;

import main.math.Tuple;
import main.util.Util;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.log;

public class Dataset {
    private String name;
    private String[][] data;                    // 2D array of entire parsed .data file (rows are examples, columns are attributes)
    private int numAttributes;                  // The number of attributes (including the target attribute)
    private int target;                         // The array index of the target attribute column in the data array
    private int type;                           // 0 = discrete, 1 = continuous
    private String[][] uniqueAttributeValues;   // 2D array of unique attribute values for each attribute (rows are attributes, columns are values)
    private List<String> classes;               // List of unique class (i.e. target attribute) values
    private List<Integer> rowIndices;           // List of the rows from the original dataset (i.e. data) that make up the current dataset (i.e. subset)
    private double entropy;                     // Entropy of dataset

    public static final int DISCRETE = 0;
    public static final int CONTINUOUS = 1;

    public Dataset(String[][] data, int target, int type) {
        List<Integer> rowIndices = IntStream.range(0, data.length).boxed().collect(Collectors.toList());
        initialize(data, rowIndices, target, type);
        indexStrings();
    }

    public Dataset(Dataset dataset, List<Integer> rowIndices) {
        initialize(dataset.getData(), rowIndices, dataset.getTarget(), dataset.getType());
        this.uniqueAttributeValues = dataset.getUniqueAttributeValues();
        this.classes = dataset.getClasses();
        this.name = dataset.name;
    }

    private void initialize(String[][] data, List<Integer> rowIndices, int target, int type) {
        this.data = data;
        this.rowIndices = rowIndices;
        this.numAttributes = data[0].length;
        this.target = target;
        this.type = type;
        this.entropy = -1;
    }

    public String[][] getData() { return data; }

    public int getTarget() { return target; }

    public int getNumAttributes() { return numAttributes; }

    public int getNumberOfExamples() { return rowIndices.size(); }

    public int getType() { return type; }

    public String[][] getUniqueAttributeValues() { return uniqueAttributeValues; }

    public List<String> getClasses() { return classes; }

    public boolean isEmpty() { return rowIndices.size() == 0; }

    public List<String> getAttributeColumn(int attribute) {
        List<String> attributes = new ArrayList<>();
        for (int row : rowIndices) {
            attributes.add(data[row][attribute]);
        }
        return attributes;
    }

    public void setName(String name) { this.name = name; }

    /**
     * Splits examples in dataset by class value
     * @return  HashMap mapping each class value to the list of examples with that class value
     */
    public Map<Integer, Dataset> splitByClass() {
        Map<Integer, List<Integer>> classRows = new HashMap<>();
        List<String[]> examples = rows();
        List<String> classes = getClasses();

        for (int i=0; i<classes.size(); i++) {
            classRows.put(i, new ArrayList<>());
        }

        for (int i = 0; i < examples.size(); i++) {
            int value = classes.indexOf(examples.get(i)[target]);
            classRows.computeIfAbsent(value, k -> new ArrayList<>());
            classRows.get(value).add(rowIndices.get(i));
        }

        Map<Integer, Dataset> separated = new HashMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : classRows.entrySet()) {
            Dataset subset = new Dataset(this, entry.getValue());
            separated.putIfAbsent(entry.getKey(), subset);
        }

        return separated;
    }

    public List<String[]> rows() {
        List<String[]> rows = new ArrayList<>();
        for (Integer row : rowIndices) {
            rows.add(data[row]);
        }
        return rows;
    }

    /**
     * Splits examples into new subsets based on values of given attribute
     * @param attribute     the attribute to split on
     * @return              a Tuple containing the unique attribute values and the split datasets
     */
    public Tuple<List<String>, List<Dataset>> splitByDiscreteAttribute(int attribute) {
        List<String> values = new ArrayList<>();
        List<Dataset> subsets = new ArrayList<>();
        Map<String,List<Integer>> attributeExamples = new HashMap<>();

        String[] attributeValues = getUniqueAttributeValues()[attribute];
        for (String value : attributeValues) {
            attributeExamples.put(value, new ArrayList<>());
        }

        List<String[]> examples = rows();
        for (int i=0; i<examples.size(); i++) {
            String value = examples.get(i)[attribute];
            attributeExamples.computeIfAbsent(value, k -> new ArrayList<>());
            attributeExamples.get(value).add(rowIndices.get(i));
        }

        for (Map.Entry<String, List<Integer>> entry : attributeExamples.entrySet()) {
            Dataset subset = new Dataset(this, entry.getValue());
            values.add(entry.getKey());
            subsets.add(subset);
        }

        return new Tuple<>(values, subsets);
    }

    /**
     * Splits examples into 2 new subsets based on the best determined "pivot" (i.e. threshold)
     * @param attribute     the attribute to split on
     * @return              a Tuple containing the pivot and the two split datasets
     */
    public Tuple<Double, Tuple<Dataset, Dataset>> splitByContinuousAttribute(int attribute) {
        double[] values = getAttributeColumn(attribute).stream().mapToDouble(Double::parseDouble).toArray();
        Arrays.sort(values);
        // Best attribute value, 2 subsets
        Tuple<Double, Tuple<Dataset, Dataset>> bestSubsets = null;

        double minimumEntropy = 0;
        for (int i = 0; i < (values.length - 1); i++) {
            Double current = values[i];
            Double next = values[i+1];
            if(current.equals(next)) {
                continue;
            }
            double pivot = (current + next) / 2;
            Tuple<Dataset, Dataset> subsets = splitAtPivot(attribute, pivot);

            double entropy = calculateWeightedEntropy(subsets);
            if(bestSubsets == null || entropy < minimumEntropy) {
                bestSubsets = new Tuple<>(pivot, subsets);
                minimumEntropy = entropy;
            }
        }
        return bestSubsets;
    }

    /**
     * Splits the examples into two subsets based on pivot value for given attribute
     * @param attribute     the attribute to split on
     * @param pivot         the pivot value (under subset contains values < pivot, over subset contains values > pivot)
     * @return              the two new split subsets
     */
    private Tuple<Dataset, Dataset> splitAtPivot(int attribute, double pivot) {
        List<Integer> lessThan = new ArrayList<>();
        List<Integer> greaterThan = new ArrayList<>();
        List<String[]> examples = rows();
        for (int i=0; i<examples.size(); i++) {
            double value = Double.parseDouble(examples.get(i)[attribute]);
            if (value <= pivot) {
                lessThan.add(rowIndices.get(i));
            } else if (value > pivot) {
                greaterThan.add(rowIndices.get(i));
            } else {
                System.err.println("Value shouldn't match pivot");
                System.exit(-1);
            }
        }

        Dataset under = new Dataset(this, lessThan);
        Dataset over = new Dataset(this, greaterThan);

        return new Tuple<>(under, over);
    }

    /**
     * Calculates the weighted entropy of the subsets (i.e. From the info gain formula on Wikipedia, this
     * calculates -sum(p(t)*H(t)) for all subsets t.
     *
     * @param subsets       the list of subsets
     * @return              the weighted entropy of the subsets
     */
    public static double calculateWeightedEntropy(List<Dataset> subsets) {
        double entropy = 0.0;
        int sum = 0;
        for(Dataset subset : subsets) {
            sum += subset.getNumberOfExamples();
            entropy += subset.entropy() * (double) subset.getNumberOfExamples();
        }

        entropy /= (sum * subsets.size());
        return entropy;
    }

    /**
     * Calculates the weighted entropy of the two subsets for continuous attributes
     *
     * @param subsets       the two subsets
     * @return              the weighted entropy of the two subsets
     */
    public static double calculateWeightedEntropy(Tuple<Dataset, Dataset> subsets) {
        double entropy = 0;
        entropy += subsets.first().entropy() * subsets.first().getNumberOfExamples();
        entropy += subsets.second().entropy() * subsets.second().getNumberOfExamples();

        entropy /= ((subsets.first().getNumberOfExamples() + subsets.second().getNumberOfExamples()) * 2);

        return entropy;
    }

    /**
     * Calculates the entropy of the dataset
     * @return      the dataset's entropy
     */
    public double entropy() {
        // If we've calculated entropy already return it
        if (entropy > 0) return entropy;
        // Entropy of an empty dataset is trivially zero
        if (isEmpty()) return 0.0;
        // Get frequency of each class value in the dataset
        Map<String, Integer> classes = Util.getFrequencyMap(getAttributeColumn(target));

        double sum = 0.0;
        for (int count : classes.values()) {
            sum += count;
        }

        entropy = 0.0;
        for (int count : classes.values()) {
            entropy -= count / sum * log2(count / sum);
        }

        return entropy;
    }

    public static double log2(double value) {
        return log(value) / log(2);
    }

    /**
     * Stores the unique attribute values and class values in uniqueAttributeValues and classes, respctively
     */
    private void indexStrings() {
        List<String[]> uniqueValues = new ArrayList<>();
        for (int col=0; col<numAttributes; col++) {
            List<String> uniqueColumn = getAttributeColumn(col).stream().distinct().collect(Collectors.toList());
            uniqueValues.add(uniqueColumn.toArray(new String[0]));
            if (col == target) {
                classes = uniqueColumn;
            }
        }
        String[][] uniqueArray = new String[uniqueValues.size()][0];
        uniqueAttributeValues = uniqueValues.toArray(uniqueArray);
    }

    @Override
    public String toString() {
        return this.name + "["+this.getNumberOfExamples()+" x "+this.getNumAttributes()+"]";
    }

}
