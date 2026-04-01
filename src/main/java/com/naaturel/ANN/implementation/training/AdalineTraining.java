package com.naaturel.ANN.implementation.training;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.Trainer;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.domain.model.training.TrainingPipeline;
import com.naaturel.ANN.implementation.adaline.AdalineTrainingContext;
import com.naaturel.ANN.implementation.gradientDescent.SquareLossStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimpleCorrectionStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimpleDeltaStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimpleErrorRegistrationStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimplePredictionStep;
import com.naaturel.ANN.infrastructure.graph.GraphVisualizer;

import java.util.List;


public class AdalineTraining implements Trainer {

    public AdalineTraining(){

    }

    @Override
    public void train(float learningRate, int epoch, Model model, DataSet dataset) {
        AdalineTrainingContext context = new AdalineTrainingContext();
        context.dataset = dataset;
        context.model = model;
        context.learningRate = learningRate;

        List<AlgorithmStep> steps = List.of(
                new SimplePredictionStep(context),
                new SimpleDeltaStep(context),
                new SquareLossStep(context),
                new SimpleErrorRegistrationStep(context),
                new SimpleCorrectionStep(context)
        );

        new TrainingPipeline(steps)
                .stopCondition(ctx -> ctx.globalLoss <= 0.00F || ctx.epoch > epoch)
                .beforeEpoch(ctx -> ctx.globalLoss = 0.0F)
                .afterEpoch(ctx -> ctx.globalLoss /= context.dataset.size())
                .withTimeMeasurement(true)
                .withVerbose(true, 1)
                .withVisualization(true, new GraphVisualizer())
                .run(context);
    }

    /*public void train(Neuron n, float learningRate, DataSet dataSet) {
        int epoch = 1;
        int maxEpoch = 202;
        float errorThreshold = 0.0F;
        float mse;

        do {
            if(epoch > maxEpoch) break;
            mse = 0;
             for(DataSetEntry entry : dataSet) {
                this.updateInputs(n, entry);
                float prediction = n.predict();
                float expectation = dataSet.getLabel(entry).getValue();
                float delta = this.calculateDelta(expectation, prediction);
                float loss = this.calculateLoss(delta);

                mse += loss;

                float currentBias = n.getBias().getWeight();
                float biasCorrector = currentBias + (learningRate * delta * n.getBias().getInput());
                n.updateBias(new Weight(biasCorrector));

                for(Synapse syn : n.getSynapses()){
                    float synCorrector = syn.getWeight() + (learningRate * delta * syn.getInput());
                    syn.setWeight(synCorrector);
                }

                System.out.printf("Epoch : %d ", epoch);
                System.out.printf("predicted : %.2f, ", prediction);
                System.out.printf("expected : %.2f, ", expectation);
                System.out.printf("delta : %.2f, ", delta);
                System.out.printf("loss : %.5f\n", loss);
            }
            mse /= dataSet.size();
            System.out.printf("[Total error : %f]\n", mse);
            System.out.println("[Final weights]");
            System.out.printf("Bias: %f\n", n.getBias().getWeight());
            int i = 1;
            for(Synapse syn : n.getSynapses()){
                System.out.printf("Syn %d: %f\n", i, syn.getWeight());
                i++;
            }
            epoch++;
        } while(mse > errorThreshold);

    }

    private void updateInputs(Neuron n, DataSetEntry entry){
        int index = 0;
        for(float value : entry){
            n.setInput(index, new Input(value));
            index++;
        }
    }

    private float calculateDelta(float expected, float predicted){
        return expected - predicted;
    }

    private float calculateLoss(float delta){
        return (float) Math.pow(delta, 2)/2;
    }*/

}
