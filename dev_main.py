"""
The purpose of this model is to be able to execute the code without installing the package
need to make this more light weight
"""

# TODO: get rid of all this code and have just an argparse main

from nlpt.data_loaders.LanguageLoader import LanguageLoader
from nlpt.modules.GRU import GRU
from nlpt.config import basic_conf
from nlpt.config import basic_hparams

config = basic_hparams.default


def main(data, rnn):
    """
    main()
    """
    losses = []
    for epoch in range(config["num_epochs"]):

        print("{} epoch: {} {}".format("=" * 20, epoch, "=" * 20))

        for i, batch in enumerate(data.sentences(config["num_batches"])):
            input_, target = batch

            loss, outputs = rnn.train(input_, target)  # .copy())
            losses.append(loss)

            if i % 100 == 0:
                print("Loss at step {}: {:.2f}".format(i, loss))
                print("Truth: \"{}\"".format(data.vec_to_sentence(target)))
                print("Guess: \"{}\"\n".format(
                    data.vec_to_sentence(outputs[:-1])))
                rnn.save()


def translate(data, rnn):
    """
    translate()
    """
    vecs = data.sentence_to_vec("the president is here <EOS>")

    translation = rnn.eval(vecs)
    print(data.vec_to_sentence(translation))


if __name__ == "__main__":
    """
    Usage: $ python main_dev.py
    """
    run_data = LanguageLoader(basic_conf.en_path, basic_conf.fr_path, config["vocab_size"], config["max_length"])
    run_rnn = GRU(run_data.input_size, run_data.output_size)
    main(run_data, run_rnn)
    # translate(data, rnn)
