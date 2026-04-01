package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.infrastructure.dataset.DataSet;

import java.util.HashMap;
import java.util.Map;

public class GradientBackpropagationContext extends TrainingContext {

    public final Map<Neuron, Float> errorSignals;
    public final float[] correctionBuffer;

    public int currentSample;
    public int batchSize;

    public GradientBackpropagationContext(Model model, DataSet dataSet, float learningRate, int batchSize){
        this.model = model;
        this.dataset = dataSet;
        this.learningRate = learningRate;
        this.batchSize = batchSize;

        this.errorSignals = new HashMap<>();
        this.correctionBuffer = new float[model.synCount()];
        this.currentSample = 1;
    }
}
