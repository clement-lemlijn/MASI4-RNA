package com.naaturel.ANN.implementation.gradientDescent;

import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.infrastructure.dataset.DataSet;

import java.util.List;

public class GradientDescentTrainingContext extends TrainingContext {

    public List<Float> correctorTerms;

    public GradientDescentTrainingContext(Model model, DataSet dataset) {
        super(model, dataset);
    }
}
