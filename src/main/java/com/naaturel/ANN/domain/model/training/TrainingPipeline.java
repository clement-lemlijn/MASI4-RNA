package com.naaturel.ANN.domain.model.training;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;
import com.naaturel.ANN.domain.model.neuron.Input;
import com.naaturel.ANN.infrastructure.graph.GraphVisualizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Predicate;

public class TrainingPipeline {

    private final List<AlgorithmStep> steps;
    private Consumer<TrainingContext> beforeEpoch;
    private Consumer<TrainingContext> afterEpoch;
    private Predicate<TrainingContext> stopCondition;

    private boolean verbose;
    private boolean visualization;
    private boolean timeMeasurement;

    private GraphVisualizer visualizer;
    private int verboseDelay;

    public TrainingPipeline(List<AlgorithmStep> steps) {
        this.steps = new ArrayList<>(steps);
        this.stopCondition = (ctx) -> false;
        this.beforeEpoch = (context -> {});
        this.afterEpoch = (context -> {});
    }

    public TrainingPipeline stopCondition(Predicate<TrainingContext> predicate) {
        this.stopCondition = predicate;
        return this;
    }

    public TrainingPipeline beforeEpoch(Consumer<TrainingContext> consumer) {
        this.beforeEpoch = consumer;
        return this;
    }

    public TrainingPipeline afterEpoch(Consumer<TrainingContext> consumer) {
        this.afterEpoch = consumer;
        return this;
    }

    public TrainingPipeline withVerbose(boolean enabled, int epochDelay) {
        if(epochDelay <= 0) throw new IllegalArgumentException("Epoch delay cannot lower or equal to 0");
        this.verbose = enabled;
        this.verboseDelay = epochDelay;
        return this;
    }

    public TrainingPipeline withVisualization(boolean enabled, GraphVisualizer visualizer) {
        this.visualization = enabled;
        this.visualizer = visualizer;
        return this;
    }

    public TrainingPipeline withTimeMeasurement(boolean enabled) {
        this.timeMeasurement = enabled;
        return this;
    }

    public void run(TrainingContext ctx) {

        long start = this.timeMeasurement ? System.currentTimeMillis() : 0;

        do {
            this.beforeEpoch.accept(ctx);
            this.executeSteps(ctx);
            this.afterEpoch.accept(ctx);
            if(this.verbose && ctx.epoch % this.verboseDelay == 0) {
                System.out.printf("[Global error] : %f\n", ctx.globalLoss);
            }
            ctx.epoch += 1;
        } while (!this.stopCondition.test(ctx));

        if(this.timeMeasurement) {
            long end = System.currentTimeMillis();
            System.out.printf("[Training finished in %.3fs]", (end-start)/1000.0);
        }

        if(this.visualization) this.visualize(ctx);
    }

    private void executeSteps(TrainingContext ctx){
        for (DataSetEntry entry : ctx.dataset) {

            ctx.currentEntry = entry;
            ctx.expectations = ctx.dataset.getLabelsAsFloat(entry);

            for (AlgorithmStep step : steps) {
                step.run();
            }

            if(this.verbose && ctx.epoch % this.verboseDelay == 0) {
                System.out.printf("Epoch : %d, ", ctx.epoch);
                System.out.printf("predicted : %s, ", Arrays.toString(ctx.predictions.toArray()));
                System.out.printf("expected : %s, ", Arrays.toString(ctx.expectations.toArray()));
                System.out.printf("delta : %s, ", Arrays.toString(ctx.deltas.toArray()));
                System.out.printf("loss : %.5f\n", ctx.localLoss);
            }
        }
    }

    private void visualize(TrainingContext ctx){
        AtomicInteger neuronIndex = new AtomicInteger(0);
        ctx.model.forEachNeuron(n -> {
            List<Float> weights = new ArrayList<>();
            n.forEachSynapse(syn -> weights.add(syn.getWeight()));

            float b  = weights.get(0);
            float w1 = weights.get(1);
            float w2 = weights.get(2);

            this.visualizer.addEquation("boundary_" + neuronIndex.getAndIncrement(), w1, w2, b, -3, 3);
        });
        int i = 0;
        for(DataSetEntry entry : ctx.dataset){
            List<Input> inputs = entry.getData();
            this.visualizer.addPoint("p"+i, inputs.get(0).getValue(), inputs.get(1).getValue());
            this.visualizer.addPoint("p"+i, inputs.get(0).getValue()+0.01F, inputs.get(1).getValue()+0.01F);
            i++;
        }
        this.visualizer.buildLineGraph();
    }

}
