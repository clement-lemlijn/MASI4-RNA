package com.naaturel.ANN.domain.abstraction;

import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;

import java.util.List;

public abstract class TrainingContext {
    public Model model;
    public DataSet dataset;
    public DataSetEntry currentEntry;

    public List<Float> expectations;
    public List<Float> predictions;
    public List<Float> deltas;

    public float globalLoss;
    public float localLoss;

    public float learningRate;
    public int epoch;
}
