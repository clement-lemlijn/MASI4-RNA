package com.naaturel.ANN.infrastructure.dataset;

import java.util.List;

public class Labels {

    private final List<Float> values;

    public Labels(List<Float> value){
        this.values = value;
    }

    public List<Float> getValues() {
        return values.stream().toList();
    }
}
