from latmats.regression import RegressionModel
from latmats.pretraining.model import Word2VecPretrainingModel


from latmats.tasks.loader import load_e_form, load_steels, load_zT, load_expt_gaps


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


    w2v_attention = Word2VecPretrainingModel(name="attention-3-64-8", use_attention=True, n_layers=3, d_model=64, n_heads=8)
    w2v_attention.compile()
    # w2v_attention.load_weights()
    w2v_attention.train(only_mat2vec=True)
    w2v_attention.save_weights()



    # w2vpm = Word2VecPretrainingModel(name="attention1by10", use_attention=True, n_layers=1, d_model=128, n_heads=10)
    # regression = RegressionModel("attention2by8", pretraining_model=w2vpm)


    # df = load_e_form()
    # regression.fit(df["composition"], df["e_form (eV/atom)"])