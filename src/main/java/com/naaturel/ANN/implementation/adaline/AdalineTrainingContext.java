package com.naaturel.ANN.implementation.adaline;

import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.infrastructure.dataset.DataSet;

public class AdalineTrainingContext extends TrainingContext {
    public AdalineTrainingContext(Model model, DataSet dataset) {
        super(model, dataset);
    }
}
