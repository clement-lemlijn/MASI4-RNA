package com.naaturel.ANN.infrastructure.dataset;

import com.naaturel.ANN.domain.model.neuron.Input;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DatasetExtractor {

    public DataSet extract(String path, int nbrLabels) {
        Map<DataSetEntry, Labels> data = new LinkedHashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");

                String[] rawInputs = Arrays.copyOfRange(parts, 0, parts.length-nbrLabels);
                String[] rawLabels = Arrays.copyOfRange(parts, parts.length-nbrLabels, parts.length);

                List<Input> inputs = new ArrayList<>();
                List<Float> labels = new ArrayList<>();

                for (String entry : rawInputs) {
                    inputs.add(new Input(Float.parseFloat(entry.trim())));
                }

                for (String entry : rawLabels) {
                    labels.add(Float.parseFloat(entry.trim()));
                }

                data.put(new DataSetEntry(inputs), new Labels(labels));
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read dataset from: " + path, e);
        }

        return new DataSet(data);
    }

}
