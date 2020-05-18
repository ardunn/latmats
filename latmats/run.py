from latmats.regression import RegressionModel
from latmats.pretraining.model import Word2VecPretrainingModel


from latmats.tasks.loader import load_e_form, load_steels, load_zT, load_expt_gaps
from latmats.tasks.tester import AlgorithmBenchmark


if __name__ == "__main__":

    # Older model without cyclical training
    # w2v_attention = Word2VecPretrainingModel(name="attention-1-128-4", use_attention=True, n_layers=1, d_model=128, n_heads=4)
    # w2v_attention.compile()
    # w2v_attention.train(only_mat2vec=True)
    # w2v_attention.save_weights()

    # regression = RegressionModel("attention-1-128-4", pretraining_model=w2v_attention)
    # df = load_e_form()
    # regression.fit(df["composition"], df["e_form (eV/atom)"])


    # Newer model, used in paper
    # w2v_attention = Word2VecPretrainingModel(name="attention-1-128-4-cylicaltrain", use_attention=True, n_layers=1, n_heads=4)
    # # w2v_attention.compile()
    # # w2v_attention.load_weights()
    # # w2v_attention.train(only_mat2vec=False)
    # # w2v_attention.save_weights()
    #
    # regression = RegressionModel(name="attention-1-128-4-cylicaltrain", pretraining_model=w2v_attention)
    #
    #
    # ab = AlgorithmBenchmark(regression)
    #
    # problem = "zT"
    # # problem = "e_form"
    #
    # gap_scores = ab.test(problem)
    #
    # ab.data[problem]["testing"]["df_predicted"].to_csv(f"comptexnet_{problem}_test.csv")
    #
    # print(gap_scores)
    #

    # 2-layer Dense network experiments
    # w2v_attention = Word2VecPretrainingModel(name="dense-2-128", n_layers=2)
    # w2v_attention.compile()
    # # w2v_attention.load_weights()
    # w2v_attention.train(only_mat2vec=True)
    # w2v_attention.save_weights()
    #
    # regression = RegressionModel(name="dense-2-128", pretraining_model=w2v_attention)
    #
    # ab = AlgorithmBenchmark(regression)
    #
    # problem = "zT"
    # # problem = "e_form"
    #
    # gap_scores = ab.test(problem)
    #
    # print(gap_scores)

    # Generating a plot of MAE vs amount of training data for two implementations
    n_samples_sizes = [100, 500, 1000, 5000, 10000, 50000]
    results = {ns: None for ns in n_samples_sizes}
    for n_samples in n_samples_sizes:
        # w2v_attention = Word2VecPretrainingModel(name="attention-1-128-4-cylicaltrain", use_attention=True, n_layers=1, n_heads=4)
        # regression = RegressionModel(name="attention-1-128-4-cylicaltrain", pretraining_model=w2v_attention)
        w2v_attention = Word2VecPretrainingModel(name="dense-2-128", n_layers=2)
        regression = RegressionModel(name="dense-2-128", pretraining_model=w2v_attention)

        ab = AlgorithmBenchmark(regression)

        problem = "e_form"
        gap_scores = ab.test(problem, n_train_samples=n_samples)
        results[n_samples] = gap_scores["mae"]
    print(results)



