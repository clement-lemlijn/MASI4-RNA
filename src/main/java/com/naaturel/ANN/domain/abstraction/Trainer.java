package com.naaturel.ANN.domain.abstraction;

import com.naaturel.ANN.infrastructure.dataset.DataSet;

public interface Trainer {
    void train(float learningRate, int epoch, Model model, DataSet dataset);
}
