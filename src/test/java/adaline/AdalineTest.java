package adaline;


import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.domain.abstraction.TrainingStep;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DatasetExtractor;
import com.naaturel.ANN.domain.model.neuron.*;
import com.naaturel.ANN.domain.model.training.TrainingPipeline;
import com.naaturel.ANN.implementation.adaline.AdalineTrainingContext;
import com.naaturel.ANN.implementation.gradientDescent.*;
import com.naaturel.ANN.implementation.simplePerceptron.SimpleCorrectionStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimpleDeltaStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimpleErrorRegistrationStep;
import com.naaturel.ANN.implementation.simplePerceptron.SimplePredictionStep;
import com.naaturel.ANN.implementation.training.steps.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AdalineTest {

    private DataSet dataset;
    private AdalineTrainingContext context;

    private List<Synapse> synapses;
    private Bias bias;
    private FullyConnectedNetwork network;

    private TrainingPipeline pipeline;

    @BeforeEach
    public void init(){
        dataset = new DatasetExtractor()
                .extract("C:/Users/Laurent/Desktop/ANN-framework/src/main/resources/assets/and-gradient.csv", 1);

        List<Synapse> syns = new ArrayList<>();
        syns.add(new Synapse(new Input(0), new Weight(0)));
        syns.add(new Synapse(new Input(0), new Weight(0)));

        bias =  new Bias(new Weight(0));

        Neuron neuron = new Neuron(syns, bias, new Linear(1, 0));
        Layer layer = new Layer(List.of(neuron));
        network = new FullyConnectedNetwork(List.of(layer));

        context = new AdalineTrainingContext();
        context.dataset = dataset;
        context.model = network;

        List<TrainingStep> steps = List.of(
                new PredictionStep(new SimplePredictionStep(context)),
                new DeltaStep(new SimpleDeltaStep(context)),
                new LossStep(new SquareLossStep(context)),
                new ErrorRegistrationStep(new SimpleErrorRegistrationStep(context)),
                new WeightCorrectionStep(new SimpleCorrectionStep(context))
        );

        pipeline = new TrainingPipeline(steps)
                .stopCondition(ctx -> ctx.globalLoss <= 0.1329F || ctx.epoch > 10000)
                .beforeEpoch(ctx -> {
                    ctx.globalLoss = 0.0F;
                });
    }

    @Test
    public void test_the_whole_algorithm(){

        List<Float> expectedGlobalLosses = List.of(
                0.501522F,
                0.498601F
        );

        context.learningRate = 0.03F;
        pipeline.afterEpoch(ctx -> {
            ctx.globalLoss /= context.dataset.size();

            int index = ctx.epoch-1;
            if(index >= expectedGlobalLosses.size()) return;

            //assertEquals(expectedGlobalLosses.get(index), context.globalLoss, 0.00001f);
        });

        pipeline.run(context);
        assertEquals(214, context.epoch);
    }
}

