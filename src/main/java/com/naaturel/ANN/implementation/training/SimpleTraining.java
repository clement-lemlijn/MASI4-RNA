package com.naaturel.ANN.implementation.training;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.Trainer;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.implementation.simplePerceptron.*;
import com.naaturel.ANN.domain.model.training.TrainingPipeline;

import java.util.List;

public class SimpleTraining implements Trainer {

    public SimpleTraining() {

    }

    @Override
    public void train(float learningRate, int epoch, Model model, DataSet dataset) {
        SimpleTrainingContext context = new SimpleTrainingContext();
        context.dataset = dataset;
        context.model = model;
        context.learningRate = learningRate;

        List<AlgorithmStep> steps = List.of(
                new SimplePredictionStep(context),
                new SimpleDeltaStep(context),
                new SimpleLossStrategy(context),
                new SimpleErrorRegistrationStep(context),
                new SimpleCorrectionStep(context)
        );

        TrainingPipeline pipeline = new TrainingPipeline(steps);
        pipeline
                .stopCondition(ctx -> ctx.globalLoss == 0.0F || ctx.epoch > epoch)
                .beforeEpoch(ctx -> ctx.globalLoss = 0)
                .withVerbose(true, 1)
                .run(context);
    }

    /*public void train(Neuron n, float learningRate, DataSet dataSet) {
        int epoch = 1;
        int errorCount;

        do {
            errorCount = 0;
            System.out.printf("Epoch : %d\n", epoch);
            for(DataSetEntry entry : dataSet) {
                this.updateInputs(n, entry);
                float prediction = n.predict();
                float expectation = dataSet.getLabel(entry).getValue();
                float delta = this.calculateDelta(expectation, prediction);
                float loss = this.calculateLoss(delta);
                if(delta > 1e-6f) {
                    this.updateWeights(n, learningRate, delta);
                    errorCount += 1;
                }
                System.out.printf("predicted : %.2f, ", prediction);
                System.out.printf("expected : %.2f, ", expectation);
                System.out.printf("delta : %.2f\n", this.calculateDelta(expectation, prediction));
            }
            System.out.print("====================================\n");
            epoch++;
        } while (errorCount != 0);
    }

    private void updateInputs(Neuron n, DataSetEntry entry){
        int index = 0;
        for(float value : entry){
            n.setInput(index, new Input(value));
            index++;
        }
    }

    private void updateWeights(Neuron n, float rate, float delta){

        Weight biasCorrection = new Weight(n.getBias().getWeight() + (rate * delta * n.getBias().getInput()));
        n.updateBias(biasCorrection);

        for(Synapse syn : n.getSynapses()){
            syn.setWeight(syn.getWeight() + (rate * delta * syn.getInput()));
        }
    }

    private float calculateDelta(float expected, float predicted){
        return expected - predicted;
    }

    private float calculateLoss(float delta){
        return Math.abs(delta);
    }
*/
}
