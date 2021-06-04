package main;

import main.math.Statistics;
import main.trees.ID3;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KFold {
    int k;
    private Dataset dataset;
    private Classifier classifier;

    public KFold(int k) {
        this.k = k;
    }

    public void init(Dataset d, Classifier c) {
        dataset = d;
        classifier = c;
    }

    static class Split {
        private final int[] trainIndices;
        private final int[] testIndices;

        public Split(int[] trainIdx, int[] testIdx) {
            trainIndices = trainIdx;
            testIndices = testIdx;
        }

        public int[] getTrainIndices() { return trainIndices; }

        public int[] getTestIndices() { return testIndices; }

    }

    private static int[] combineTrainFolds(int[][] folds, int totalSize, int excludeIndex) {
        int size = totalSize - folds[excludeIndex].length;
        int[] result = new int[size];

        int start = 0;
        for (int i = 0; i < folds.length; i++) {
            if (i == excludeIndex) {
                continue;
            }
            int[] fold = folds[i];
            System.arraycopy(fold, 0, result, start, fold.length);
            start = start + fold.length;
        }

        return result;
    }

    public static void shuffle(int[] array) {
        int index;
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            if (index != i) {
                array[index] ^= array[i];
                array[i] ^= array[index];
                array[index] ^= array[i];
            }
        }
    }

    public List<Split> createKFolds(int[] indices) {
        int[][] foldIndexes = new int[k][];

        int step = indices.length / k;
        int beginIndex = 0;

        for (int i = 0; i < k-1; i++) {
            foldIndexes[i] = Arrays.copyOfRange(indices, beginIndex, beginIndex + step);
            beginIndex = beginIndex + step;
        }

        foldIndexes[k-1] = Arrays.copyOfRange(indices, beginIndex, indices.length);

        List<Split> folds = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            int[] testIdx = foldIndexes[i];
            int[] trainIdx = combineTrainFolds(foldIndexes, indices.length, i);
            folds.add(new Split(trainIdx, testIdx));
        }

        return folds;
    }

    public List<Double> crossValidate() {
        // Step 1: Shuffle data (this shuffles the indices instead for better performance)
        int numberOfExamples = dataset.getNumberOfExamples();
        int[] indices = IntStream.range(0, numberOfExamples).toArray();
        shuffle(indices);

        // Step 2: Split into k folds
        List<Split> folds = createKFolds(indices);

        // Step 3: For each k-fold, train using other folds and test using single fold
        List<Double> accuracies = new ArrayList<>();
        List<String> classes = dataset.getClasses();
        for (Split fold: folds) {
            int[] trainIdx = fold.getTrainIndices();
            Dataset trainDataset = new Dataset(dataset, Arrays.stream(trainIdx).boxed().collect(Collectors.toList()));
            int[] testIdx = fold.getTestIndices();
            Dataset testDataset = new Dataset(dataset, Arrays.stream(testIdx).boxed().collect(Collectors.toList()));
            // We use the test dataset for post pruning for ID3 (but not for ensemble methods)
            if (classifier instanceof ID3) ((ID3) classifier).setPruningDataset(testDataset);

            classifier.train(trainDataset);
            String[] predictions = new String[testDataset.getNumberOfExamples()];
            List<String[]> examples = testDataset.rows();
            for (int i = 0; i < examples.size(); i++) {
                String[] example = examples.get(i);
                int prediction = classifier.classify(example);
                String predictedClass = classes.get(prediction);
                predictions[i] = predictedClass;
            }
            accuracies.add(Statistics.computeAccuracy(examples, predictions, dataset.getTarget()));
        }

        return accuracies;
    }


}
