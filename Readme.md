# CoronAI: A Machine Learning Toolkit for Coronavirus

An __ERLab__ Project

### Introdcution
Please view this [article](https://medium.com/@shayan.fazeli/coronai-deep-information-extraction-for-covid-19-related-articles-f7c016149586?sk=80cd4fb635854934083f2c2dd26b66b2) in that regard.

### Installation
Clone this github repository and run:
```bash
python3 setup.py install
```

### Usage
Please find sample experiment commands below:

##### Segment to Vector
To generate the mathematical vectors for our text segments, the following command can be used:

```bash
coronai_segment2vector --gpu=2 --input_csv=.../text_segment_dataset.csv \
--output_pkl=.../output.pkl \
--path_to_bert_weights=.../NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16 \
--batch_size=25
```

This batch size works fine on a 2080Ti GPU. Note that part of this code is responsible for preprocessing 
and storing AllenNLP compatible instances to be fed to the BERT-based pipeline. 
There is no need to create them more than once, so if the `input_text_sequence_instances.pkl` file exists
in the folder of your `output_pkl` file, it will be used directly. You can also
get it from our shared drive, along with some representations and other files.

##### Unsupervised Segment Group Discovery

To do this, you need the mathematical representations of course. When you have those, the following
command should be used:

```bash
coronai_unsupervised_clustering --method_name=kmeans \
--method_params=[random_state:int:2019] --method_searchspace=n_clusters:5:100 \
--output_bundle=...path_to_output_folder/kmeans.pkl \
--input_files=...path_to/full_representations_dataset.pkl

```

```bash
coronai_unsupervised_clustering --method_name=birch \
--method_params=[branching_factor:int:50,threshold:float:0.5] \
--method_searchspace=n_clusters:3:100 \
--output_bundle=...path_to_output_folder/birch.pkl \
--input_files=...path_to/full_representations_dataset.pkl
```

The method searchspace allows you to choose one parameter and do a search over it. In our case, we use number of clusters.
In every step, the output file will be updated (so you do not need to wait forever to be able to check things).
Also, the format for inputting method parameters are obvious.

### Documentation

The documentation for this project is available in [here](http://web.cs.ucla.edu/~shayan/coronai/docs/)

```bash
username: user
password: mlisLife
```

### CoronAI Shared Drive

Please view [here](https://drive.google.com/drive/folders/1NGwteuPIbX3acWRLASFnos3lnhS9XgPk?usp=sharing)