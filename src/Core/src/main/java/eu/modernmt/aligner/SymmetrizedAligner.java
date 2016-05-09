package eu.modernmt.aligner;

import eu.modernmt.aligner.symal.Symmetrization;
import eu.modernmt.model.Sentence;
import eu.modernmt.processing.AlignmentsInterpolator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.concurrent.*;

/**
 * Created by davide on 09/05/16.
 */
public class SymmetrizedAligner implements Aligner {

    public static final Symmetrization.Strategy DEFAULT_SYMMETRIZATION_STRATEGY = Symmetrization.Strategy.GrowDiagFinalAnd_ORIGINAL;

    private final Logger logger = LogManager.getLogger(getClass());
    private final Aligner forwardModel;
    private final Aligner backwardModel;
    private Symmetrization.Strategy symmetrizationStrategy;

    public SymmetrizedAligner(Aligner forwardModel, Aligner backwardModel) {
        this.forwardModel = forwardModel;
        this.backwardModel = backwardModel;
        this.symmetrizationStrategy = DEFAULT_SYMMETRIZATION_STRATEGY;
    }

    @Override
    public void load() throws AlignerException {
        ExecutorService pool = Executors.newFixedThreadPool(2);

        Future<Void> forward = pool.submit(new LoadModelTask(forwardModel, true));
        Future<Void> backward = pool.submit(new LoadModelTask(backwardModel, true));

        try {
            forward.get();
            backward.get();
        } catch (InterruptedException e) {
            throw new AlignerException("Loading interrupted", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();

            if (cause instanceof AlignerException)
                throw (AlignerException) cause;
            else
                throw new AlignerException("Unexpected exception while loading models", cause);
        } finally {
            pool.shutdownNow();
        }
    }

    @Override
    public int[][] getAlignments(Sentence sentence, Sentence translation) throws AlignerException {
        int[][] forwardAlignments = forwardModel.getAlignments(sentence, translation);
        int[][] backwardAlignments = backwardModel.getAlignments(sentence, translation);

        int[][] alignments = Symmetrization.symmetrizeAlignment(forwardAlignments, backwardAlignments, symmetrizationStrategy);

        return AlignmentsInterpolator.interpolateAlignments(alignments, sentence.getWords().length, translation.getWords().length);
    }

    public void setSymmetrizationStrategy(Symmetrization.Strategy strategy) {
        this.symmetrizationStrategy = strategy;
    }

    @Override
    public void close() throws IOException {
        try {
            forwardModel.close();
        } finally {
            backwardModel.close();
        }
    }

    private class LoadModelTask implements Callable<Void> {

        private final Aligner model;
        private final boolean forward;

        public LoadModelTask(Aligner model, boolean forward) {
            this.model = model;
            this.forward = forward;
        }

        @Override
        public Void call() throws AlignerException {
            logger.info(String.format("Loading %s model: %s", (forward ? "forward" : "backward"), model.getClass().getSimpleName()));

            long time = System.currentTimeMillis();
            model.load();
            long elapsed = System.currentTimeMillis() - time;

            logger.info(String.format("%s model loaded in %.1f", (forward ? "Forward" : "Backward"), elapsed / 1000.));

            return null;
        }
    }

}
