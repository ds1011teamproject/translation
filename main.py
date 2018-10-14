from LanguageLoader import *
from GRU import *

import settings

config = settings.default


def main(data, rnn):
    """
    main()
    """
    losses = []
    for epoch in range(config["num_epochs"]):

        print("{} epoch: {} {}".format("=" * 20, epoch, "=" * 20))

        for i, batch in enumerate(data.sentences(config["num_batches"])):
            input_, target = batch

            loss, outputs = rnn.train(input_, target.copy())
            losses.append(loss)

            if (i % 100 == 0):
                print("Loss at step {}: {:.2f}".format(i, loss))
                print("Truth: \"{}\"".format(data.vec_to_sentence(target)))
                print("Guess: \"{}\"\n").format(
                    data.vec_to_sentence(outputs[:-1]))
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
    Usage: $ python main.py
    """
    data = LanguageLoader(settings.en_path, settings.fr_path,
                          config["vocab_size"], config["max_length"])
    rnn = GRU(data.input_size, data.output_size)

    main(data, rnn)
    # translate(data, rnn)
