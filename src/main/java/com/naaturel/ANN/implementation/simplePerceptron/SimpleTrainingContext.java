package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.infrastructure.dataset.DataSet;

public class SimpleTrainingContext extends TrainingContext {
    public SimpleTrainingContext(Model model, DataSet dataset) {
        super(model, dataset);
    }
}
