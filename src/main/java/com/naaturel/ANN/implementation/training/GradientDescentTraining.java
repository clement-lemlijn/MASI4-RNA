package com.naaturel.ANN.implementation.training;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.Trainer;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.implementation.gradientDescent.GradientDescentErrorStrategy;
import com.naaturel.ANN.implementation.gradientDescent.GradientDescentTrainingContext;
import com.naaturel.ANN.domain.model.training.TrainingPipeline;
import com.naaturel.ANN.implementation.gradientDescent.GradientDescentCorrectionStrategy;
import com.naaturel.ANN.implementation.gradientDescent.SquareLossStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimpleDeltaStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimplePredictionStep;
import com.naaturel.ANN.infrastructure.graph.GraphVisualizer;

import java.util.ArrayList;
import java.util.List;

public class GradientDescentTraining implements Trainer {

    public GradientDescentTraining(){

    }

    @Override
    public void train(float learningRate, int epoch, Model model, DataSet dataset) {
        GradientDescentTrainingContext context = new GradientDescentTrainingContext(model, dataset);
        context.learningRate = learningRate;
        context.correctorTerms =  new ArrayList<>();

        List<AlgorithmStep> steps = List.of(
                new SimplePredictionStep(context),
                new SimpleDeltaStep(context),
                new SquareLossStep(context),
                new GradientDescentErrorStrategy(context)
        );

        new TrainingPipeline(steps)
            .stopCondition(ctx -> ctx.globalLoss <= 0.08F || ctx.epoch > epoch)
            .beforeEpoch(ctx -> {
                GradientDescentTrainingContext gdCtx = (GradientDescentTrainingContext) ctx;
                gdCtx.globalLoss = 0.0F;
                gdCtx.correctorTerms.clear();
                for(int i = 0; i < gdCtx.model.synCount(); i++){
                    gdCtx.correctorTerms.add(0F);
                }
            })
            .afterEpoch(ctx -> {
                context.globalLoss /= context.dataset.size();
                new GradientDescentCorrectionStrategy(context).run();
            })
            //.withVerbose(true)
            .withTimeMeasurement(true)
            .withVisualization(true, new GraphVisualizer())
            .run(context);
    }

    /*public void train(Neuron n, float learningRate, DataSet dataSet) {
        int epoch = 1;
        int maxEpoch = 402;
        float errorThreshold = 0F;
        float mse;

        do {
            if(epoch > maxEpoch) break;

            float biasCorrector = 0;
            mse = 0;
            List<Float> correctorTerms = this.initCorrectorTerms(n.getSynCount());

            for(DataSetEntry entry : dataSet) {
                this.updateInputs(n, entry);
                float prediction = n.predict();
                float expectation = dataSet.getLabel(entry).getValue();
                float delta = this.calculateDelta(expectation, prediction);
                float loss = this.calculateLoss(delta);

                mse += loss;

                biasCorrector += learningRate * delta * n.getBias().getInput();

                for(int i = 0; i < correctorTerms.size(); i++){
                    Synapse syn = n.getSynapse(i);
                    float c = correctorTerms.get(i);
                    c += learningRate * delta * syn.getInput();
                    correctorTerms.set(i, c);
                }

                System.out.printf("Epoch : %d ", epoch);
                System.out.printf("predicted : %.2f, ", prediction);
                System.out.printf("expected : %.2f, ", expectation);
                System.out.printf("delta : %.2f, ", delta);
                System.out.printf("loss : %.2f\n", loss);
            }
            mse /= dataSet.size();
            System.out.printf("[Total error : %f]\n", mse);

            float currentBias = n.getBias().getWeight();
            float newBias = currentBias + biasCorrector;
            n.updateBias(new Weight(newBias));

            for(int i = 0; i < correctorTerms.size(); i++){
                Synapse syn = n.getSynapse(i);
                float c = syn.getWeight() + correctorTerms.get(i);
                syn.setWeight(c);
            }

            epoch++;
        } while(mse > errorThreshold);

        System.out.println("[Final weights]");
        System.out.printf("Bias: %f\n", n.getBias().getWeight());
        int i = 1;
        for(Synapse syn : n.getSynapses()){
            System.out.printf("Syn %d: %f\n", i, syn.getWeight());
            i++;
        }
    }

    private List<Float> initCorrectorTerms(int number){
        List<Float> res = new ArrayList<>();
        for(int i = 0; i < number; i++){
            res.add(0F);
        }
        return res;
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
    }

    public float computeThreshold(DataSet dataSet) {
        float sum = 0;
        for (DataSetEntry entry : dataSet) {
            sum += dataSet.getLabel(entry).getValue();
        }
        float mean = sum / dataSet.size();

        float variance = 0;
        for (DataSetEntry entry : dataSet) {
            float diff = dataSet.getLabel(entry).getValue() - mean;
            variance += diff * diff;
        }
        variance /= dataSet.size();

        return variance;
    }*/

}
