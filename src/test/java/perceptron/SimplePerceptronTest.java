package perceptron;

import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.domain.abstraction.TrainingStep;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DatasetExtractor;
import com.naaturel.ANN.domain.model.neuron.*;
import com.naaturel.ANN.domain.model.training.TrainingPipeline;
import com.naaturel.ANN.implementation.simplePerceptron.*;
import com.naaturel.ANN.implementation.training.steps.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;


public class SimplePerceptronTest {

    private DataSet dataset;
    private SimpleTrainingContext context;

    private List<Synapse> synapses;
    private Bias bias;
    private FullyConnectedNetwork network;

    private TrainingPipeline pipeline;

    @BeforeEach
    public void init(){
        dataset = new DatasetExtractor()
                .extract("C:/Users/Laurent/Desktop/ANN-framework/src/main/resources/assets/and.csv", 1);

        List<Synapse> syns = new ArrayList<>();
        syns.add(new Synapse(new Input(0), new Weight(0)));
        syns.add(new Synapse(new Input(0), new Weight(0)));

        bias =  new Bias(new Weight(0));

        Neuron neuron = new Neuron(syns, bias, new Heaviside());
        Layer layer = new Layer(List.of(neuron));
        network = new FullyConnectedNetwork(List.of(layer));

        context = new SimpleTrainingContext();
        context.dataset = dataset;
        context.model = network;

        List<TrainingStep> steps = List.of(
                new PredictionStep(new SimplePredictionStep(context)),
                new DeltaStep(new SimpleDeltaStep(context)),
                new LossStep(new SimpleLossStrategy(context)),
                new ErrorRegistrationStep(new SimpleErrorRegistrationStep(context)),
                new WeightCorrectionStep(new SimpleCorrectionStep(context))
        );

        pipeline = new TrainingPipeline(steps);
        pipeline.stopCondition(ctx -> ctx.globalLoss == 0.0F || ctx.epoch > 100);
        pipeline.beforeEpoch(ctx -> ctx.globalLoss = 0);
    }

    @Test
    public void test_the_whole_algorithm(){

        List<Float> expectedGlobalLosses = List.of(
                2.0F,
                3.0F,
                3.0F,
                2.0F,
                1.0F,
                0.0F
        );

        context.learningRate = 1F;
        pipeline.afterEpoch(ctx -> {
            int index = ctx.epoch-1;
            assertEquals(expectedGlobalLosses.get(index), context.globalLoss);
        });

        pipeline.run(context);
        assertEquals(6, context.epoch);
    }
}
