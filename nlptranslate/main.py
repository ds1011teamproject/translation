from nlptranslate.data_loaders.LanguageLoader import LanguageLoader
from nlptranslate.modules.GRU import GRU
from nlptranslate.config import basic_settings

config = basic_settings.default


def main(data, rnn):
    """
    main()
    """
    losses = []
    for epoch in range(config["num_epochs"]):

        print("{} epoch: {} {}".format("=" * 20, epoch, "=" * 20))

        for i, batch in enumerate(data.sentences(config["num_batches"])):
            """
            TODO:
            - data.sentences needs to be a Tensor instead of a list of tensors
            - after that is fixed, call input_.cuda() and target.cuda()
            """
            input_, target = batch

            loss, outputs = rnn.train(input_, target)  # .copy())
            losses.append(loss)

            if (i % 100 == 0):
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
    Usage: $ python main.py
    """
    data = LanguageLoader(basic_settings.en_path, basic_settings.fr_path,
                          config["vocab_size"], config["max_length"])
    rnn = GRU(data.input_size, data.output_size)  # TODO: .cuda()

    main(data, rnn)
    # translate(data, rnn)
