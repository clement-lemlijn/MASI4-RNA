package com.naaturel.ANN.implementation.training;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.Trainer;
import com.naaturel.ANN.domain.model.training.TrainingPipeline;
import com.naaturel.ANN.implementation.gradientDescent.SquareLossStep;
import com.naaturel.ANN.implementation.multiLayers.BackpropagationCorrectionStep;
import com.naaturel.ANN.implementation.multiLayers.GradientBackpropagationContext;
import com.naaturel.ANN.implementation.multiLayers.ErrorSignalStep;
import com.naaturel.ANN.implementation.multiLayers.OutputLayerErrorStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimplePredictionStep;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import java.util.List;

public class GradientBackpropagationTraining implements Trainer {
    @Override
    public void train(float learningRate, int epoch, Model model, DataSet dataset) {
        GradientBackpropagationContext context =
                new GradientBackpropagationContext(model, dataset, learningRate, dataset.size());

        List<AlgorithmStep> steps = List.of(
                new SimplePredictionStep(context),
                new OutputLayerErrorStep(context),
                new ErrorSignalStep(context),
                new BackpropagationCorrectionStep(context),
                new SquareLossStep(context)
        );

        new TrainingPipeline(steps)
                .stopCondition(ctx -> ctx.globalLoss <= 0.00F || ctx.epoch > epoch)
                .beforeEpoch(ctx -> {
                    ctx.globalLoss = 0.0F;
                })
                .afterEpoch(ctx -> {
                    ctx.globalLoss /= dataset.size();
                })
                .withVerbose(false,epoch/10)
                .withTimeMeasurement(true)
                .run(context);
    }
}
