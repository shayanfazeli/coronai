#!/usr/bin/env python

__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'
"""
    CoronAI: Unsupervised Clustering
    ==========
    This application is provided to assist researchers with easy and efficient clustering of comment databases
    using coronai's data structs. (The application is consistent with my other erlab project: DiagNote)
"""
# libraries
from typing import Dict, Any, List
import argparse
import os
import numpy
import pandas
import pickle, json
import matplotlib.pyplot as plt
import plotnine as p9
from tqdm import tqdm
import sklearn
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans
import numpy.linalg as linear_algebra


def parse_parameters_from_special_string(input_string: str) -> Dict[str, Any]:
    """
    [Function has been taken from diagnote project]

    The :func:`parse_parameters_from_special_string` returns the dict version of the optimization parameters for PyTorch optimizers.
    Note that the `params` argument is unattended and has to be filled later.

    Parameters
    ----------
    input_string: `str`, required
        The string format of parameter, type, value. For example: `lr:float:0.001,weight_decay:float:0.1`.

    Returns
    ----------
    The `Dict[str, Any]` of the parameters and their values.
    """
    optimization_parameters = dict()
    optimization_parameters_string_list = input_string[1:-1].split(',')
    for element in optimization_parameters_string_list:
        element_name, element_type, element_value = element.split(':')
        if element_type == "int":
            element_value = int(element_value)
        elif element_type == "float":
            element_value = float(element_value)
        elif element_type == "str":
            element_value = str(element_value)
        elif element_type == "list":
            element_value = [int(e) for e in element_value[1:-1].split(';')]
        else:
            raise ValueError
        optimization_parameters[element_name] = element_value

    return optimization_parameters


def compute_point_loss(X: numpy.ndarray, labels: List[int], average: bool = True) -> float:
    """
    The :func:`compute_point_loss` method can be used for computing the point-based loss in our
    unsupervised scheme

    Parameters
    ----------
    X: `numpy.ndarray`, required
        The input

    labels: `List[int]`, required
        The labels that are associated with inputs by the inference scheme

    average: `bool`, optional (default=True)
        Boolean indicator to determine whether or not the loss is to be averaged.

    Returns
    ----------
    The `float` loss value.
    """
    centroids = dict()
    counts = dict()
    for i in range(len(labels)):
        if not labels[i] in centroids.keys():
            centroids[labels[i]] = X[i, :].ravel()
            counts[labels[i]] = 1
        else:
            centroids[labels[i]] += X[i, :].ravel()
            counts[labels[i]] += 1
    for key in centroids.keys():
        centroids[key] /= float(counts[key])
    del counts
    loss = 0
    for i in range(len(labels)):
        loss += linear_algebra.norm(X[i, :].ravel() - centroids[labels[i]])
    if average:
        loss /= float(len(labels))
    return loss


def main(args: argparse.Namespace):
    """
    The main function of this application. Sample command to run this application is:

    ```
    coronai_unsupervised_clustering --method_name=kmeans --method_params=[random_state:int:2019] --method_searchspace=n_clusters:5:100 --output_bundle=/home/shayan/data/outputs/coronai/kmeans.pkl --input_files=/home/shayan/data/outputs/coronai/full_representations_dataset.pkl

    coronai_unsupervised_clustering --method_name=birch --method_params=[branching_factor:int:50,threshold:float:0.5] --method_searchspace=n_clusters:3:100 --output_bundle=/home/shayan/data/outputs/coronai/birch.pkl --input_files=/home/shayan/data/outputs/coronai/full_representations_dataset.pkl

    ```

    Parameters
    ----------
    args: `argparse.Namespace`, required
        The argument namespace passed from terminal
    """
    if args.method_name == 'kmeans':
        kmeans_pipeline(args)
    elif args.method_name == 'birch':
        birch_pipeline(args)
    else:
        raise ValueError


def fetch_input_dict(input_files: str):
    """
    The :func:`fetch_input_dict` is responsible for fetching the input file.

    Parameters
    ----------
    input_files: `str`, required
        The filepath for the input dataset
    """
    if input_files.startswith('['):
        input_files = input_files[1:-1].split(',')

    # defining input dict
    input_dict = {
        'dataset_index': [],
        'text_segment': [],
        'vector_representation': [],
        'paper_id': [],
        'corresponding_section': []}

    if isinstance(input_files, List):
        for input_file in input_files:
            with open(os.path.abspath(input_file), 'rb') as handle:
                tmp_dict = pickle.load(handle)
                for key in input_dict.keys():
                    input_dict[key] += tmp_dict[key]
                del tmp_dict
    else:
        with open(os.path.abspath(input_files), 'rb') as handle:
            input_dict = pickle.load(handle)

    return input_dict


def kmeans_pipeline(args: argparse.Namespace) -> None:
    """
    The main function of this application when kmeans is requested.

    Parameters
    ----------
    args: `argparse.Namespace`, required
        The argument namespace passed from terminal
    """
    method_parameters = parse_parameters_from_special_string(args.method_params)
    input_dict = fetch_input_dict(input_files=args.input_files)
    X = numpy.array(input_dict['vector_representation'])
    method_searchspace = args.method_searchspace
    variable_name, variable_low, variable_high = method_searchspace.split(':')
    variable_low = int(variable_low)
    variable_high = int(variable_high)
    history = []

    output_filepath = os.path.abspath(args.output_bundle)

    if args.batch_size == 0:
        for i in tqdm(range(variable_low, variable_high + 1)):
            method_parameters[variable_name] = i
            method_model = KMeans(**method_parameters).fit(X)
            method_model.fit(X)

            history.append(
                {'search_on': variable_name,
                 'method_name': 'kmeans',
                 'parameters': method_parameters,
                 'input_files': args.input_files,
                 'labels': method_model.labels_,
                 'loss': compute_point_loss(X=X, labels=method_model.labels_)
                 }
            )

            with open(output_filepath, 'wb') as handle:
                pickle.dump({'history': history}, handle)

            del method_model
    else:
        for i in tqdm(range(variable_low, variable_high + 1)):
            method_parameters[variable_name] = i
            method_model = MiniBatchKMeans(**method_parameters)

            random_index_permutation = numpy.random.permutation(X.shape[0])
            labels = numpy.zeros(X.shape[0])

            for epoch in range(args.num_epochs):
                print('>> (status): epoch {}/{}\n'.format(epoch, args.num_epochs))
                cursor = 0
                while (cursor + args.batch_size) <= X.shape[0]:
                    method_model.partial_fit(
                        X[random_index_permutation[cursor:(cursor+args.batch_size)], :])
                    labels[random_index_permutation[cursor:(cursor+args.batch_size)]] = method_model.labels_
                    cursor += args.batch_size

            history.append(
                {'search_on': variable_name,
                 'method_name': 'kmeans',
                 'parameters': method_parameters,
                 'input_files': args.input_files,
                 'labels': labels,
                 'loss': compute_point_loss(X=X, labels=method_model.labels_)
                 }
            )

            with open(output_filepath, 'wb') as handle:
                pickle.dump({'history': history}, handle)

            del method_model

    print('\n>> status: all finished.\n')


def birch_pipeline(args: argparse.Namespace) -> None:
    """
    The main function of this application when birch is requested.

    Parameters
    ----------
    args: `argparse.Namespace`, required
        The argument namespace passed from terminal
    """
    method_parameters = parse_parameters_from_special_string(args.method_params)
    input_dict = fetch_input_dict(input_files=args.input_files)
    X = numpy.array(input_dict['vector_representation'])
    method_searchspace = args.method_searchspace
    variable_name, variable_low, variable_high = method_searchspace.split(':')
    variable_low = int(variable_low)
    variable_high = int(variable_high)
    history = []

    output_filepath = os.path.abspath(args.output_bundle)

    if args.batch_size == 0:
        for i in tqdm(range(variable_low, variable_high + 1)):
            method_parameters[variable_name] = i

            if method_parameters['n_clusters'] == 0:
                method_parameters['n_clusters'] = None

            method_model = Birch(**method_parameters).fit(X)
            method_model.fit(X)

            history.append(
                {'search_on': variable_name,
                 'method_name': 'birch',
                 'parameters': method_parameters,
                 'input_files': args.input_files,
                 'labels': method_model.labels_,
                 'loss': compute_point_loss(X=X, labels=method_model.labels_)
                 }
            )

            with open(output_filepath, 'wb') as handle:
                pickle.dump({'history': history}, handle)

            del method_model
    else:
        for i in tqdm(range(variable_low, variable_high + 1)):
            method_parameters[variable_name] = i
            method_model = Birch(**method_parameters)

            random_index_permutation = numpy.random.permutation(X.shape[0])
            labels = numpy.zeros(X.shape[0])

            for epoch in range(args.num_epochs):
                print('>> (status): epoch {}/{}\n'.format(epoch, args.num_epochs))
                cursor = 0
                while (cursor + args.batch_size) <= X.shape[0]:
                    method_model.partial_fit(
                        X[random_index_permutation[cursor:(cursor + args.batch_size)], :])
                    labels[random_index_permutation[cursor:(cursor + args.batch_size)]] = method_model.labels_
                    cursor += args.batch_size

            history.append(
                {'search_on': variable_name,
                 'method_name': 'kmeans',
                 'parameters': method_parameters,
                 'input_files': args.input_files,
                 'labels': labels,
                 'loss': compute_point_loss(X=X, labels=method_model.labels_)
                 }
            )

            with open(output_filepath, 'wb') as handle:
                pickle.dump({'history': history}, handle)

            del method_model

    print('\n>> status: all finished.\n')


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser(description="CoronAI unsupervised clustering")
    parser.add_argument(
        '--input_files', type=str, required=True,
        help="The list of filenames, should either be a filename or [filepath1,filepath2]"
    )
    parser.add_argument(
        '--output_bundle', type=str, required=True,
        help='The output file is going to be a pickle file, including the status, error, and labels'
    )
    parser.add_argument(
        '--method_name', type=str, required=True,
        help='The methodname could be kmeans, birch'
    )
    parser.add_argument(
        '--method_params', type=str, required=True,
        help='The method parameters, the same as optimization parameters format of string.'
    )
    parser.add_argument(
        '--method_searchspace', type=str, required=True,
        help='The format is: num_clusters:1:50,random_state:1:20'
    )

    parser.add_argument(
        '--max_memory', type=int, default=30000,
        help='The maximum amount of memory that the application is allowed to use, in MB'
    )

    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='The maximum amount of memory that the application is allowed to use, in MB'
    )

    parser.add_argument(
        '--batch_size', type=int, default=0,
        help='If a positive int is provided, will be used to do the unsupervised clustering with mini-batches'
    )

    args = parser.parse_args()

    # running the main app
    with sklearn.config_context(working_memory=args.max_memory):
        main(args)
