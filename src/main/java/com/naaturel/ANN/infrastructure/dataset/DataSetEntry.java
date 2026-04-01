package com.naaturel.ANN.infrastructure.dataset;

import com.naaturel.ANN.domain.model.neuron.Input;

import java.util.*;

public class DataSetEntry implements Iterable<Input> {

    private List<Input> data;

    public DataSetEntry(List<Input> data){
        this.data = data;
    }

    public List<Input> getData() {
        return new ArrayList<>(data);
    }


    @Override
    public int hashCode() {
        return Objects.hash(this.data);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof DataSetEntry dataSetEntry)) return false;
        return Objects.equals(this.data, dataSetEntry.data);
    }

    @Override
    public Iterator<Input> iterator() {
        return this.data.iterator();
    }


    @Override
    public String toString() {
        return Arrays.toString(this.data.toArray());
    }
}
