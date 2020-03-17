__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'

import os
from typing import Dict, Any
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from overrides import overrides


class BertSequencePooler(Seq2VecEncoder):
    def forward(
            self,
            embeddings: torch.Tensor,
            mask: torch.Tensor = None
    ):
        """
        The module `BertSequencePooler` will allow the aggregation of sequence representations.

        Parameters
        ----------
        embeddings: `torch.Tensor`, required
            The embeddings will be a tensor from which the first sequence element will be extracted.
        mask: `torch.Tensor`, optional (default=None)
            The mask that will be inputted according to the Seq2Vec base class, but wont be used in this instance.

        Returns
        ----------
        The sequence representation tensor will be returned which contains the first element only. Later on, it is
        possible to utilize the masks in a more efficient way.
        """

        # we need only the First (0-indexed) column for all the batches (the first :)
        return embeddings[:, 0]

    @overrides
    def get_output_dim(self) -> int:
        """
        This method will not be used in this case, merely overriding it to comply with the criteria.
        If the output dimension is needed, the main embedding module can be used to output that. There is no need
        to use this.
        """
        pass


def get_bert_token_indexers(path_to_bert_weights: str, maximum_number_of_tokens: int, is_model_lowercase: bool = True) -> PretrainedBertIndexer:
    """
    Retrieving bert based token indexers (which will do sub-word tokenizing)
    
    Parameters
    ----------
    path_to_bert_weights: `str`, required
        The path to the pytorch bert weights

    maximum_number_of_tokens: `int`, required
        The maximum number of tokens (truncation or sliding window based on allennlp design)

    is_model_lowercase: `bool` optional (default=True)
        Force lower casing the input to the model

    Returns
    ----------
    Returns the required instance of :class:`PretrainedBertIndexer`.
    """

    token_indexers = PretrainedBertIndexer(
        pretrained_model=os.path.abspath(
            os.path.join(path_to_bert_weights, 'vocab.txt')
        ),
        max_pieces=maximum_number_of_tokens,
        do_lowercase=is_model_lowercase,
        use_starting_offsets=True
    )

    return token_indexers


def get_bert_token_embeddings(
        path_to_bert_weights: str,
        top_layer_only: bool,
        indexer_id: str,
        token_to_embed: str = 'word',
        fine_tune_embeddings: bool = False) -> TextFieldEmbedder:
    """
    To get the BERT-based token embeddings (including NCBI-BERT, BioBERT, Standard BERT, etc. the
    :func:`get_bert_token_embeddings` can be used.
    
    Parameters
    ----------
    path_to_bert_weights: `str`, required
        The path to the pytorch bert weights

    top_layer_only: `bool`, required
        This boolean value will indicate whether only the top layer outputs should be used

    fine_tune_embeddings: `bool`, optional (default=False)
        Not useful in this project, remains here because of consistency with diagnote. Mainly
        for setting the gradients and fine-tune capability.

    token_to_embed: `str`, optional (default=word)
        Focus on embedding words or subwords

    indexer_id: `str`, required
        The indexer id for allennlp

    Returns
    ----------
    It returns the `TextFieldEmbedder` object for embedding.
    """

    path_to_model = path_to_bert_weights
    top_layer_only = top_layer_only
    token_to_embed = token_to_embed
    fine_tune_embeddings = fine_tune_embeddings

    if token_to_embed == 'word':
        embedder_to_indexer_map = {indexer_id: {
            'input_ids': indexer_id,
            'offsets': indexer_id + '-offsets'
        }}
        assert isinstance(embedder_to_indexer_map, dict), "it should be a dict"
    else:
        embedder_to_indexer_map = None

    # preparing the self token embedding for word/subword bert tokens:
    token_embedding: TextFieldEmbedder = BasicTextFieldEmbedder(
        {
            indexer_id: PretrainedBertEmbedder(
                pretrained_model=path_to_model,
                top_layer_only=top_layer_only,
                requires_grad=fine_tune_embeddings
            )
        },
        embedder_to_indexer_map=embedder_to_indexer_map,
        allow_unmatched_keys=True
    )

    return token_embedding
