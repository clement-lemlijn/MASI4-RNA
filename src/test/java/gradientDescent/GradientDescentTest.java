package gradientDescent;

import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.domain.abstraction.TrainingStep;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DatasetExtractor;
import com.naaturel.ANN.domain.model.neuron.*;
import com.naaturel.ANN.domain.model.training.TrainingPipeline;
import com.naaturel.ANN.implementation.gradientDescent.*;
import com.naaturel.ANN.implementation.simplePerceptron.*;
import com.naaturel.ANN.implementation.training.steps.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;


public class GradientDescentTest {

    private DataSet dataset;
    private GradientDescentTrainingContext context;

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

        context = new GradientDescentTrainingContext();
        context.dataset = dataset;
        context.model = network;
        context.correctorTerms = new ArrayList<>();

        List<TrainingStep> steps = List.of(
                new PredictionStep(new SimplePredictionStep(context)),
                new DeltaStep(new SimpleDeltaStep(context)),
                new LossStep(new SquareLossStep(context)),
                new ErrorRegistrationStep(new GradientDescentErrorStrategy(context))
        );

        pipeline = new TrainingPipeline(steps)
                .stopCondition(ctx -> ctx.globalLoss <= 0.125F || ctx.epoch > 100)
                .beforeEpoch(ctx -> {
                    GradientDescentTrainingContext gdCtx = (GradientDescentTrainingContext) ctx;
                    gdCtx.globalLoss = 0.0F;
                    gdCtx.correctorTerms.clear();
                    for (int i = 0; i < ctx.model.synCount(); i++){
                        gdCtx.correctorTerms.add(0F);
                    }
                });
    }

    @Test
    public void test_the_whole_algorithm(){

        List<Float> expectedGlobalLosses = List.of(
                0.5F,
                0.38F,
                0.3176F,
                0.272096F,
                0.237469F
        );

        context.learningRate = 0.2F;
        pipeline.afterEpoch(ctx -> {
            context.globalLoss /= context.dataset.size();
            new GradientDescentCorrectionStrategy(context).run();

            int index = ctx.epoch-1;
            if(index >= expectedGlobalLosses.size()) return;

            assertEquals(expectedGlobalLosses.get(index), context.globalLoss, 0.00001f);
        });

        pipeline
                .withVerbose(true)
                .run(context);
        assertEquals(67, context.epoch);
    }
}
