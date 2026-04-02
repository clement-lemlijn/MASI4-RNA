package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.infrastructure.dataset.DataSet;

import java.util.HashMap;
import java.util.Map;

public class GradientBackpropagationContext extends TrainingContext {

    public final float[] errorSignals;
    public final float[] correctionBuffer;
    public final boolean[] errorSignalsComputed;

    public int currentSample;
    public int batchSize;

    public GradientBackpropagationContext(Model model, DataSet dataSet, float learningRate, int batchSize){
        super(model, dataSet);
        this.learningRate = learningRate;
        this.batchSize = batchSize;

        this.errorSignals = new float[model.neuronCount()];
        this.correctionBuffer = new float[model.synCount()];
        this.errorSignalsComputed = new boolean[model.neuronCount()];
        this.currentSample = 1;
    }
}
