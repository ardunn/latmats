from latmats.regression import RegressionModel
from latmats.pretraining.model import Word2VecPretrainingModel


from latmats.tasks.loader import load_e_form, load_steels, load_zT, load_expt_gaps
from latmats.tasks.tester import AlgorithmBenchmark


if __name__ == "__main__":

    # w2v = Word2VecPretrainingModel(name="refactor_dense", n_layers=2)
    # w2v.compile()
    # w2v.load_weights()
    # w2v.train()
    # w2v.save_weights()

    # w2v_attention = Word2VecPretrainingModel(name="attention2by8", use_attention=True, n_layers=1, n_heads=10)
    # w2v_attention.compile()
    # w2v_attention.load_weights()
    # w2v_attention.train(only_mat2vec=True)
    # w2v_attention.save_weights()


    # w2v_attention = Word2VecPretrainingModel(name="attention-1-128-4", use_attention=True, n_layers=1, d_model=128, n_heads=4)
    # w2v_attention.compile()
    # w2v_attention.train(only_mat2vec=True)
    # w2v_attention.save_weights()

    # regression = RegressionModel("attention-1-128-4", pretraining_model=w2v_attention)
    # df = load_e_form()
    # regression.fit(df["composition"], df["e_form (eV/atom)"])



    w2v_attention = Word2VecPretrainingModel(name="attention-1-128-4-cylicaltrain", use_attention=True, n_layers=1, n_heads=4)
    # w2v_attention.compile()
    # w2v_attention.load_weights()
    # w2v_attention.train(only_mat2vec=False)
    # w2v_attention.save_weights()


    regression = RegressionModel(name="attention-1-128-4-cylicaltrain", pretraining_model=w2v_attention)


    ab = AlgorithmBenchmark(regression)

    problem = "zT"
    # problem = "e_form"

    gap_scores = ab.test(problem)

    ab.data[problem]["testing"]["df_predicted"].to_csv(f"comptexnet_{problem}_test.csv")

    print(gap_scores)

    # problem = "e_form"
    # df_train = ab.data["testing"][problem]["df_train"]
    # df_test = ab.data["testing"][problem]["df_test"]

    # regression.fit(df_train["composition"], df_train["e_form (eV/atom)"])

