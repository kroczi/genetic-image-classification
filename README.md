# Genetic Classification of Images
 
Feature Extraction and Image Classification using Genetic Programming combined with Histograms of Oriented Gradients
 
## Prerequisites
The project requires Python2 or Python3 interpreter as well as appropriate version of following libraries:
 - [numpy](http://www.numpy.org/)
 - [deap](http://deap.gel.ulaval.ca/doc/default/index.html)
 - [pypng](https://pypi.python.org/pypi/pypng)
 - [termcolor](https://pypi.python.org/pypi/termcolor)
 
In order to run the application properly there has to be a modification made in the source of the `DEAP` library. The `generate` function located in the `gp.py` module of the `DAEP` package has to replaced with the contents of the [`gp_generate_modification.py`](utils/gp_generate_modification.py) file.
 
## Usage
### Binary classification
##### Configuration
In the [`config`](config/) directory there are located two `.ini` files. These are configuration files one concerning the analised dataset and the other containing the genetic algorithm parameters. User can specify their own profiles, taking pattern from the existing ones, and set the four variables in the [`binary_classifier_evaluator.py`](binary/binary_classifier_evaluator.py) file:
```Python
    DATASET_CONFIG_FILE = 'config/dataset_config.ini'
    PARAMETERS_CONFIG_FILE = 'config/parameters_config.ini'
    DATASET_PROFILE = ["<DATASET_NAME_1", "DATASET_NAME_2"]
    PARAMETERS_PROFILE = ["DATASET_NAME_1_PARAMETERS", "DATASET_NAME_2_PARAMETERS"]
```
##### Data preparation
It is required to configure data sets in the [`dataset-config`](config/dataset_config.ini) file.
```Python
    [<DATASET_NAME>]
    min_width = <IMAGES_WIDTH>
    min_height = <IMAGES_HEIGHT>
    base_directory = <DATASET_DIR>/
    training_directory = <TRAINING_SUBDIR>/
    testing_directory = <TESTING_SUBDIR>/
    classes = <NUMBER_OF_CLASSES>
```
The data set with 10 classes should have the following form. It is also required that the collection consists of png images.
```Bash
├── testing
│   ├── 0
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ....
│   ├── 1
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ...
│   ├── 9
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ...
├── training
│   ├── 0
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ....
│   ├── 1
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ...
│   ├── ...
│   ├── 9
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ...
```
##### Execution
To perform the process of creating binary classifiers user needs to execute following command from the main directory:
```
python -m binary.binary_classifier_evaluator
```
There should be created two log files. One of them - the `info.log` will contain binary classifiers.
### Multiclass classification
##### Data preparation
Apart from the images required for the binary classification there has to be created a file containing binary classifiers for each pair of classes. It should contain one classifier in each line. Optionally, after the classifier there can be comma-separated priority of the classifier (e.g. learn-rate). If the priority is not provided then all classifiers are of equal importance with the priority set to one. Additionally the classifiers have to be sorted with respect to their classes.
##### Configuration
In the appropriate file in the [`multiclass`](multiclass/) directory there have to be set following variables:
```Python
    CLASSES = <NUMBER_OF_CLASSES>
    CLASSIFIERS_FILE = <FILE_WITH_BINARY_CLASSIFIERS>
    BASE_DIRECTORY = <DATASET_DIR>
```
##### Execution
Depending on which method of multiclass classification the user wants to use, they need to execute following commands from the main directory:
 - _One versus one_
    ```
    python -m multiclass.one_versus_one_classfier
    ```
 - _All  pairs filter tree_
    ```
    python -m multiclass.all_pairs_filter_tree_classifier
    ```

