__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'


from typing import Union, List
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


def preprocess_text(
        text: str,
        mode: int = 1,
        verbose: bool = False,
        return_sentences: bool = False,
        split_words_in_sentences: bool = False,
        force_lower_case: bool = True) -> Union[str, List[str], List[List[str]]]:
    """
    The :func:`preprocess_text` helps with providing an easy-to-use interface for the basic pre-processing
    routines for NLP applications. Since most of the times our input is in text format, however unclean, using
    this method we can manage to prepare a better input for the tokenizer to peruse.

    Parameters
    ---------
    text: `str`, required
        The input which is in a string format and will be preprocessed using this method.
    mode: `int`, optional (default=2)
        This is the mode of pre-processing. When it is `0` the same text is going to be the output, which
        is designed for specific cases for modular design, when whether or not the system is going to be
        used with applying pre-processing.
        Currently, we have a universal way of preprocessing which is mode `1`.
    verbose: `bool`, optional (default=False)
        This parameter decides whether or not to print the outputs indicating the progress and status
        of pre-processing to the standard output.
    return_sentences: `bool`, optional (default=False)
        If this parameter is set to true, the output will be the pre-processed sentences. The sentence
        tokenization is performed using `nltk.tokenize.sent_tokenize` and it happens after pre-processing
        the whole text.
    split_words_in_sentences: `bool`, optional (default=False)
        If we have `return_sentences=True`, having this parameter set as `True` will command the pre-processor
        to return a `List[List]` in which each `List` represents the words of a sentence.
    force_lower_case: `bool`, optional (default=True)
        If you are working with a cased version of an embedding, set this to false. Usually, in the tokenizer,
        the text lowercasing is handled. However, having it here as well can help the user with getting a better
        grasp of how the data will look like in the pre-processing stage.

    Returns
    ----------
    The output is the pre-processed text of type `Union[str, List[str], List[List[str]]]`.
    """
    modes = [
        'No Change',
        'General cleaning of the text, substituting special characters with their names, pruning punctuations, preserving dots',
    ]

    if force_lower_case:
        text = text.lower()

    if verbose:
        print('\n>> The opration for the selected mode {} is: {}\n'.format(mode, modes[mode]))

    if mode == 0:
        return text
    elif mode == 1:
        # first, finding the a.b type numbers.
        text = re.sub(r'[0-9]+.[0-9]+', ' number ', text)

        # now, finding normal numbers:
        text = re.sub(r'[0-9]+', ' number ', text)

        # taking care of some special characters
        text = re.sub(r'&', ' and ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'@', ' at ', text)
        text = re.sub(r'=', ' equals ', text)
        text = re.sub(r'%', ' percent ', text)
        text = re.sub(r'\$', ' dollar ', text)
        text = text.replace('+', ' plus ')
        text = re.sub(r'-', ' minus ', text)

        # replacing comma with dot
        text = re.sub(r'[:;.?!]+', '.', text)

        # the rest, put a space just to make sure some tokens are separated and not resulting in unnecessary OOV.
        text = re.sub(r'[*,)(/]+', ' ', text)
        text = re.sub(r'[\n]+', ' ', text)
        text = re.sub(r'[\r]+', ' ', text)
        text = re.sub(r'[\t]+', ' ', text)

        # now just some last enforcing
        text = re.sub(r'[^a-zA-Z. ]+', '', text).strip()

        # removing unnecessary spaces.
        text = text.replace(".", ". ")
        text = re.sub(r'[ ]+', ' ', text).strip()
        text = text.replace(" .", ".")

        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    else:
        raise ValueError("Bad mode value is received by the preprocessing engine.")

    if not return_sentences:
        text = text.replace('.', '. ')
        text = re.sub(r'[ ]+', ' ', text).strip()
        return text
    else:
        if not split_words_in_sentences:
            sentences = [re.sub(r'[.]+', '', e) for e in sent_tokenize(text)]
        else:
            sentences = [re.sub(r'[.]+', '', e).split(' ') for e in sent_tokenize(text)]
        return sentences
