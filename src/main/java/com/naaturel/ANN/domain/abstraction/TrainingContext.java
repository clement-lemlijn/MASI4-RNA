package com.naaturel.ANN.domain.abstraction;

import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;

import java.util.List;

public abstract class TrainingContext {
    public Model model;
    public DataSet dataset;
    public DataSetEntry currentEntry;

    public List<Float> expectations;
    public float[] predictions;
    public float[] deltas;

    public float globalLoss;
    public float localLoss;

    public float learningRate;
    public int epoch;

    public TrainingContext(Model model, DataSet dataset) {
        this.model = model;
        this.dataset = dataset;
        this.deltas = new float[dataset.getNbrLabels()];
    }

}
