#!/usr/bin/env python
__title__ = 'Project CoronAI'
__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'

# libraries
import argparse
import pickle
import numpy
from tqdm import tqdm
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import PassThroughIterator, BucketIterator
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ListField, ArrayField, MetadataField
import pandas
import os
from coronai.library.anlp import get_bert_token_indexers, get_bert_token_embeddings, BertSequencePooler
from coronai.library.preprocessing import preprocess_text


def main(args):
    """
    CoronAI: Sequence2Vector
    ==========

    In many comment analysis and note analysis applications it is important to be able to represent portions
    of the string as numeric vectors. This application in CoronAI project assists us with that. Using this application,
    one can read the clean CSV file that includes the text segments, and represent each and every one of them
    using a variant of BERT.

    Sample command to run:

    ```
    coronai_segment2vector --gpu=2 --input_csv=.../text_segment_dataset.csv --output_pkl=.../output.pkl --path_to_bert_weights=.../NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16 --batch_size=400
    ```

    Parameters
    ----------
    args: `args.Namespace`, required
        The arguments needed for it
    """

    # reading the input files

    input_dataframe = pandas.read_csv(os.path.abspath(args.input_csv))

    # todo: currently the system is for bert only. we will add arguments to control this and make
    # it modular for other embeddings such as elmo, etc.

    # todo: the paranthesis of wordpiece_tokenizer covering was an issue found in other places as well,
    # make sure to look for them and resolve the issue.
    source_token_indexers = get_bert_token_indexers(path_to_bert_weights=args.path_to_bert_weights, maximum_number_of_tokens=512, is_model_lowercase=True)
    source_tokenizer = lambda x: source_token_indexers.wordpiece_tokenizer(
        preprocess_text(x))[:510]

    source_token_embeddings = get_bert_token_embeddings(
        path_to_bert_weights=args.path_to_bert_weights,
        top_layer_only=True,
        indexer_id='source_tokens',
        token_to_embed='word',
    )

    # finding the output dimension for the token embeddings, to be used later.
    source_embeddings_dimension = source_token_embeddings.get_output_dim()

    sequence_encoder = BertSequencePooler()

    input_text_sequence_instances = []

    print(">> (status): preparing data...\n\n")
    for i in tqdm(range(input_dataframe.shape[0])):
        fields = dict()
        row = input_dataframe.iloc[i, :]
        tokens = [Token(x) for x in source_tokenizer(row['text_segment'])]
        sequence_field = TextField(tokens, {'source_tokens': source_token_indexers})
        fields['dataset_index'] = MetadataField(i)
        fields['source_tokens'] = sequence_field
        fields['paper_id'] = MetadataField(row['paper_id'])
        fields['text_segment'] = MetadataField(row['text_segment'])
        fields['corresponding_section'] = MetadataField(row['corresponding_section'])
        input_text_sequence_instances.append(
            Instance(fields)
        )

    with open(os.path.join(os.path.basename(args.output_pkl), 'input_text_sequence_instances.pkl'), 'wb') as handle:
        pickle.dump(input_text_sequence_instances, handle)

    print(">> (info): the input_text_sequence_instances file is successfully saved in the storage.\n")

    # now it's time to encode the now tokenized instances.
    batch_size = args.batch_size
    #iterator = PassThroughIterator()
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[('source_tokens', 'num_tokens')])
    vocabulary = Vocabulary()
    iterator.index_with(vocabulary)

    number_of_instances = len(input_text_sequence_instances)
    data_stream = iterator(iter(input_text_sequence_instances))

    output_corpora = dict()
    output_corpora['dataset_index'] = list()
    output_corpora['text_segment'] = list()
    output_corpora['vector_representation'] = list()
    output_corpora['paper_id'] = list()
    output_corpora['corresponding_section'] = list()

    for batches_processed_sofar in tqdm(range(0, number_of_instances // batch_size + 1)):
        sample = next(data_stream)

        vector_representations = sequence_encoder(
            source_token_embeddings(sample['source_tokens'])
        ).data.cpu().numpy()

        output_corpora['text_segment'] += sample['text_segment']
        output_corpora['paper_id'] += sample['paper_id']
        output_corpora['corresponding_section'] += sample['corresponding_section']
        output_corpora['dataset_index'] += sample['dataset_index']
        output_corpora['vector_representation'] += [numpy.array(e) for e in vector_representations.tolist()]

        if batches_processed_sofar > 0 and batches_processed_sofar % 10 == 0:
            if not os.path.isdir(os.path.join(os.path.basename(args.output_pkl), 'batches')):
                os.makedirs(os.path.join(os.path.basename(args.output_pkl), 'batches'))
            with open(os.path.join(os.path.basename(args.output_pkl), 'batches/batch_{}.pkl'.format(batches_processed_sofar)), 'wb') as handle:
                pickle.dump(input_text_sequence_instances, handle)

    with open(args.output_pkl, 'wb') as handle:
        pickle.dump(output_corpora, handle)

    print('>> (info): all done.\n')


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser(description="CoronAI segment2vector")

    parser.add_argument(
        '--batch_size', type=int, default=128,
        help="The batch size to process"
    )
    parser.add_argument(
        '--input_csv', type=str, required=True,
        help='This parameter indicates the path to the input csv file. This file should contain the column text_segment'
    )
    parser.add_argument(
        '--output_pkl', type=str, required=True,
        help='The output file is going to be a pickle file, including the sequences and their corresponding representations.'
    )
    parser.add_argument(
        '-p', '--path_to_bert_weights', type=str, required=True,
        help="The path to the corresponding bert weights (pytorch weights) to be used."
    )
    parser.add_argument(
        '-g', '--gpu', type=int, default=-1, help="preferred GPU id. If set as -1, the process automatically chooses the GPU with highest memory availability.") # -1: selecting best gpu is automatic based on MEMORY availability

    args = parser.parse_args()

    # running the main app
    main(args)

