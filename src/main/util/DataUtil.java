package main.util;

import main.Dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import static main.Main.*;

public class DataUtil {
    private static final String dataFolder = "./data/";

    public static Dataset parseFile(String fileName) throws IOException {
        String filePath = dataFolder + fileName;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
            List<String[]> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                List<String> attributes = new ArrayList<>(Arrays.asList(line.split("[\\s,]+")));
                // Remove 0.48, 0.50 columns from ecoli
                if (fileName.equals("ecoli.data")) {
                    attributes.remove(4);
                    attributes.remove(3);
                }
                // Remove ID attribute
                if (fileName.equals("breast-cancer-wisconsin.data") || fileName.equals("ecoli.data")) {
                    attributes.remove(0);
                }
                lines.add(attributes.toArray(new String[0]));
            }
            String[][] array = new String[lines.size()][0];
            String[][] data = lines.toArray(array);

            int targetColumn;
            if (fileName.equals(mushroomData) || fileName.equals(letterRecognitionData)) {
                targetColumn = 0;
            } else {
                targetColumn = data[0].length - 1;
            }
            // 0 = discrete attributes, 1 = continuous attributes
            int type = 0;
            if (fileName.equals(ecoliData) || fileName.equals(letterRecognitionData)) type = 1;
            if (fileName.equals(mushroomData) || fileName.equals(breastCancerData)) {
                DataUtil.replaceMissingValues(array);
            }

            return new Dataset(array, targetColumn, type);
        } catch (IOException e) {
            throw new IOException("Error while parsing data file.");
        }
    }

    public static void replaceMissingValues(String[][] data) {
        for (int i=0; i<data.length; i++) {
            for (int j=0; j<data[i].length; j++) {
                if (data[i][j].equals("?")) {
                    data[i][j] = getMostFrequentAttribute(data, j);
                }
            }
        }
    }

    private static String getMostFrequentAttribute(String[][] data, int attribute) {
        HashMap<String,Integer> frequencyMap = new HashMap<>();
        int max = 1;
        String temp = "";
        for (String[] row : data) {
            String attr = row[attribute];
            if (frequencyMap.get(attr) != null && !attr.equals("?")) {
                int count = frequencyMap.get(attr);
                count++;
                frequencyMap.put(attr, count);
                if (count > max) {
                    max = count;
                    temp = attr;
                }
            } else {
                frequencyMap.put(attr, 1);
            }
        }
        return temp;
    }
}
