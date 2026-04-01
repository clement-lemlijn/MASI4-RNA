package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SimpleDeltaStep implements AlgorithmStep {

    private final TrainingContext context;

    public SimpleDeltaStep(TrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        DataSet dataSet = context.dataset;
        DataSetEntry entry = context.currentEntry;
        List<Float> predicted = context.predictions;
        List<Float> expected = dataSet.getLabelsAsFloat(entry);

        //context.delta = label.getValue() - context.predictions;
        context.deltas = IntStream.range(0, predicted.size())
                .mapToObj(i -> expected.get(i) - predicted.get(i))
                .collect(Collectors.toList());
    }

}
