import argparse
from environment import Environment
from lstm_par import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary, load_vocabulary

VOCABULARY = 'typos/voc.txt'
RESTORE = 'typos/checkpoints/final'
DATASET = 'typos/all_scipop.txt'


def predict(string, vocabulary=VOCABULARY, dataset=DATASET, restore=RESTORE):
    print('BEGIN' + string + 'END')
    if vocabulary is None:
        if dataset is not None:
            with open(dataset, 'r') as f:
                text = f.read()
            vocabulary = create_vocabulary(text)
    else:
        vocabulary = load_vocabulary(vocabulary)
    vocabulary_size = len(vocabulary)
    # print('(typos.predict)vocabulary_size:', vocabulary_size)
    # print('(typos.predict)vocabulary:\n', vocabulary)

    env = Environment(Lstm, LstmBatchGenerator, vocabulary=vocabulary)

    valid_add_feed = [# {'placeholder': 'sampling_prob', 'value': 1.},
                      {'placeholder': 'dropout', 'value': 1.}]


    env.build(batch_size=64,
              num_layers=2,
              num_nodes=[1300, 1300],
              num_output_layers=2,
              num_output_nodes=[2048],
              vocabulary_size=vocabulary_size,
              embedding_size=512,
              num_unrollings=100,
              init_parameter=3.,
              regime='inference',
              num_gpus=1)

    _, example_res = env.test(
        restore_path=restore,
        additions_to_feed_dict=valid_add_feed,
        validation_dataset_texts=[string],
        printed_result_types=[],
        example_length=len(string),
        vocabulary=vocabulary,
        print_results=False,
        verbose=False
    )
    return example_res[0]['input'][1:], example_res[0]['output'][1:], example_res[0]['prob_vecs'][1:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file2parse', help='path to file which will be processed by language model', default=None
    )
    parser.add_argument(
        '-s', '--string', help='string to be parsed by language model', default=None
    )
    parser.add_argument(
        "-v", "--vocabulary", help="path to file with vocabulary corresponding to loaded model", default=VOCABULARY)
    parser.add_argument("-r", "--restore", help="path to file with checkpoint", default=RESTORE)
    parser.add_argument(
        "-d", "--dataset", help="path to dataset which will be used for vocabulary creation", default=DATASET)
    parser.add_argument(
        "-p", "--probabilities", help="if specified print probabilities", action='store_true')
    args = parser.parse_args()

    if args.file2parse is not None:
        with open(args.file2parse, 'r') as f:
            string = f.read()
    elif args.string is not None:
        string = args.string

    inp, out, prob_vecs = predict(string, vocabulary=args.vocabulary,
                          dataset=args.dataset, restore=args.restore)

    print('@input:\n' + inp)
    print('@output:\n' + out)
    if args.probabilities:
        print('@prob_vecs:\n', prob_vecs)