package com.naaturel.ANN.infrastructure.dataset;

import com.naaturel.ANN.domain.model.neuron.Input;

import java.util.*;
import java.util.stream.Stream;

public class DataSet implements Iterable<DataSetEntry>{

    private final Map<DataSetEntry, Labels> data;

    private final int nbrInputs;
    private final int nbrLabels;

    public DataSet() {
        this(new LinkedHashMap<>()); //ensure iteration order is the same as insertion order
    }

    public DataSet(Map<DataSetEntry, Labels> data){
        this.data = data;
        this.nbrInputs = this.calculateNbrInput();
        this.nbrLabels = this.calculateNbrLabel();
    }

    private int calculateNbrInput(){
        //assumes every entry are the same length
        Stream<DataSetEntry> keyStream = this.data.keySet().stream();
        Optional<DataSetEntry> firstEntry = keyStream.findFirst();
        return firstEntry.map(inputs -> inputs.getData().size()).orElse(0);
    }

    private int calculateNbrLabel(){
        //assumes every label are the same length
        Stream<DataSetEntry> keyStream = this.data.keySet().stream();
        Optional<DataSetEntry> firstEntry = keyStream.findFirst();
        return firstEntry.map(inputs -> this.data.get(inputs).getValues().size()).orElse(0);
    }


    public int size() {
        return data.size();
    }

    public int getNbrInputs() {
        return this.nbrInputs;
    }

    public int getNbrLabels(){
        return this.nbrLabels;
    }

    public List<DataSetEntry> getData(){
        return new ArrayList<>(this.data.keySet());
    }

    public List<Float> getLabelsAsFloat(DataSetEntry entry){
        return this.data.get(entry).getValues();
    }

    public DataSet toNormalized() {
        List<DataSetEntry> entries = this.getData();

        float maxAbs = entries.stream()
                .flatMap(e -> e.getData().stream())
                .map(Input::getValue)
                .map(Math::abs)
                .max(Float::compare)
                .orElse(1.0F);

        Map<DataSetEntry, Labels> normalized = new HashMap<>();
        for (DataSetEntry entry : entries) {
            List<Input> normalizedData = new ArrayList<>();

            for (Input input : entry.getData()) {
                Input normalizedInput = new Input(Math.round((input.getValue() / maxAbs) * 100.0F) / 100.0F);
                normalizedData.add(normalizedInput);
            }

            normalized.put(new DataSetEntry(normalizedData), this.data.get(entry));
        }

        return new DataSet(normalized);
    }

    @Override
    public Iterator<DataSetEntry> iterator() {
        return this.data.keySet().iterator();
    }
}
