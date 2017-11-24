import glob
import sys
import os
import re
import json
import shutil
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext(appName="animals_identification")

fs_config = {
    "root_directory": "/home/hadoopuser/projects/project2/distributed_learning/animals_identification/",
    "label_directory": "label/",
    "one_vs_one_model_directory": "one_model",
    "one_vs_one_label_filename": "one_file.json",
    "one_vs_all_model_directory": "all_model",
    "one_vs_all_label_filename": "all_file.json",
    "feature_directory": "features/",
    "train_one_feature_filename": "train_one_features.txt",
    "test_one_feature_filename": "test_one_features.txt",
    "train_all_feature_filename": "train_all_features.txt",
    "test_all_feature_filename": "test_all_features.txt",
}


def make_label_point_from_list(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(20000.0, values)


def make_labeled_point(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])


def split_sets(input_dir, file_pattern, number_for_train):
    if not os.path.exists(input_dir) and not os.path.isdir(input_dir):
        print("Cannot access feature files directory: {}!".format(input_dir))
        sys.exit(-1)

    train_set = {}
    test_set = {}

    directory_list = glob.glob(input_dir + '/' + file_pattern)

    for featured_file in directory_list:
        file_name = os.path.basename(featured_file)
        key = file_name[0:re.search("\d", file_name).start() - 1]

        if train_set.get(key) is None:
            train_set[key] = []

        if len(train_set[key]) >= number_for_train:
            if test_set.get(key) is None:
                test_set[key] = []
            test_set[key].append(file_name)
        else:
            train_set[key].append(file_name)

    return train_set, test_set


def create_features(train_set, test_set, input_dir, config, class1, class2):
    # Create the feature directory is it doesn't exists
    if not os.path.exists(config['root_directory'] + config['feature_directory']):
        os.mkdir(config['root_directory'] + config['feature_directory'], 0o700)

    # Create the label directory
    if not os.path.exists(config['root_directory'] + config['label_directory']):
        os.mkdir(config['root_directory'] + config['label_directory'], 0o700)

    one_vs_one = False
    if len(class1) > 0 and len(class2) > 0:
        one_vs_one = True
        label_path = config['root_directory'] + config['label_directory'] + config['one_vs_one_label_filename']
        train_feature_path = config['root_directory'] + config['feature_directory'] + \
                             config['train_one_feature_filename']
        test_feature_path = config['root_directory'] + config['feature_directory'] + config['test_one_feature_filename']
    else:
        label_path = config['root_directory'] + config['label_directory'] + config['one_vs_all_label_filename']
        train_feature_path = config['root_directory'] + config['feature_directory'] + \
                             config['train_all_feature_filename']
        test_feature_path = config['root_directory'] + config['feature_directory'] + config['test_all_feature_filename']

    # Delete train est test feature files is they exists
    if os.path.exists(train_feature_path):
        os.remove(train_feature_path)
    if os.path.exists(test_feature_path):
        os.remove(test_feature_path)

    # Remove the label file if it exists
    if os.path.exists(label_path):
        os.remove(label_path)

    # Generate the labels
    if one_vs_one:
        keys = [class1, class2]
    else:
        keys = train_set.keys()

    labels = {}
    index = 0
    for key in keys:
        labels[index] = key
        index += 1

    # Save the labels
    with open(label_path, "w") as label_file:
        json.dump(labels, label_file)

    # Generate the feature files
    for key in labels.keys():
        for file_name in train_set[labels[key]]:
            with open(input_dir + '/' + file_name, 'r') as json_feature:
                data = json.load(json_feature)
                data = str(key) + ',' + ",".join(map(str, data))
                with open(train_feature_path, "a") as feature:
                    print(data, file=feature)
    for key in labels.keys():
        for file_name in test_set[labels[key]]:
            with open(input_dir + '/' + file_name, 'r') as json_feature:
                data = json.load(json_feature)
                data = str(key) + ',' + ",".join(map(str, data))
                with open(test_feature_path, "a") as feature:
                    print(data, file=feature)


def create_model(config, class1, class2):
    # Load training data
    if len(class1) > 0 and len(class2) > 0:
        train_feature_path = config['root_directory'] + config['feature_directory'] + config[
            'train_one_feature_filename']
    else:
        train_feature_path = config['root_directory'] + config['feature_directory'] + config[
            'train_all_feature_filename']
    data = sc.textFile(train_feature_path)
    parsed_data = data.map(make_labeled_point)

    # Build the model
    model = SVMWithSGD.train(parsed_data, iterations=100)

    # Evaluate the model on training data
    labels_and_preds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
    train_err = labels_and_preds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsed_data.count())
    print("Training Error = " + str(train_err * 100) + "%")

    if len(class1) > 0 and len(class2) > 0:
        model_path = config['root_directory'] + config['one_vs_one_model_directory']
    else:
        model_path = config['root_directory'] + config['one_vs_all_model_directory']

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Save the model
    model.save(sc, model_path)


def test_all(config):
    test_feature_path = config['root_directory'] + config['feature_directory'] + config['test_all_feature_filename']
    model_path = config['root_directory'] + config['one_vs_all_model_directory']
    label_path = config['root_directory'] + config['label_directory'] + config['one_vs_all_label_filename']

    if not os.path.exists(test_feature_path):
        print('No feature for all test')
        sys.exit(-1)

    if not os.path.exists(model_path):
        print('No model for all test')
        sys.exit(-1)

    if not os.path.exists(label_path):
        print('No label for all test')
        sys.exit(-1)

    # Load test data
    data = sc.textFile(test_feature_path)
    parsed_data = data.map(make_labeled_point)
    # Load the model
    model = SVMModel.load(sc, model_path)
    # Try the model against test data
    labels_and_preds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
    test_err = labels_and_preds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsed_data.count())
    print("Test Error = " + str(test_err * 100) + "%")


def test_one_vs_one(config, class1, class2):
    test_feature_path = config['root_directory'] + config['feature_directory'] + config['test_one_feature_filename']
    model_path = config['root_directory'] + config['one_vs_one_model_directory']
    label_path = config['root_directory'] + config['label_directory'] + config['one_vs_one_label_filename']

    if not os.path.exists(test_feature_path):
        print('No feature for all test')
        sys.exit(-1)

    if not os.path.exists(model_path):
        print('No model for all test')
        sys.exit(-1)

    if not os.path.exists(label_path):
        print('No label for all test')
        sys.exit(-1)

    with open(label_path, "r") as label_file:
        labels = dict(json.load(label_file))

    label_values = labels.values()

    if not any(x == class1 for x in label_values) or not any(x == class2 for x in label_values):
        print("No labels for {} and {}.".format(class1, class2))
        sys.exit(-1)

    # Load test data
    data = sc.textFile(test_feature_path)
    parsed_data = data.map(make_labeled_point)
    # Load the model
    model = SVMModel.load(sc, model_path)
    # Try the model against test data
    labels_and_preds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
    test_err = labels_and_preds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsed_data.count())
    print("Test Error = " + str(test_err * 100) + "%")


def check_image(config, json_file):
    model_path = config['root_directory'] + config['one_vs_all_model_directory']
    label_path = config['root_directory'] + config['label_directory'] + config['one_vs_all_label_filename']

    if not os.path.exists(model_path):
        print('No model with the full set')
        sys.exit(-1)

    if not os.path.exists(label_path):
        print('No label for with the full set')
        sys.exit(-1)

    with open(label_path, "r") as label_file:
        labels = dict(json.load(label_file))

    # inv_labels = {v: k for k, v in labels.items()}

    """
    rdd = sc.textFile(json_file)\
        .map(lambda line: ((line[1:])[:-1]))\
        .flatMap(make_float_list)

    model = SVMModel.load(sc, model_path)

    label = model.predict(rdd)
    """

    with open(json_file) as features_file:
        features = json.load(features_file)

    lp = make_label_point_from_list(",".join(map(str, features)))
    model = SVMModel.load(sc, model_path)
    label = model.predict(lp.features)

    print("=" * 60)
    print("Image features file: {}".format(json_file))
    print("Label: {}".format(label))
    print("Prediction: {}".format(labels[str(label)]))
    print("=" * 60)


def usage(prog):
    print("Usage: " + prog + " [--dir <input_directory> --size <train_size_by_class> | [--test | "
                             "--check <feature_file.json>]] [--1vs1 <class1,class2> | --all]")


def main():
    i = 0
    args = sys.argv[1:]
    if len(args) < 1:
        usage(sys.argv[0])
        sys.exit(-1)

    input_dir = ""
    size = 0
    test_data = False
    check_data = ""
    class1 = ""
    class2 = ""
    all_data = False
    # Get command line parameters
    while i < len(args):
        if args[i] == "--dir":
            i += 1
            input_dir = args[i]
            if len(input_dir) == 0:
                print("Input directory name is required")
                sys.exit(-1)
        elif args[i] == "--size":
            i += 1
            if not args[i].isdigit():
                print("Training size ({}) must be a number".format(args[i]))
                sys.exit(-1)
            else:
                if int(args[i]) <= 0:
                    print("Training size ({}) must be greater than zero".format(args[i]))
                    sys.exit(-1)
                else:
                    size = int(args[i])
        elif args[i] == "--test":
            test_data = True
        elif args[i] == "--check":
            i += 1
            check_data = args[i]
        elif args[i] == "--1vs1":
            i += 1
            class1 = args[i].split(',')[0].strip()
            class2 = args[i].split(',')[1].strip()
        elif args[i] == "--all":
            all_data = True

        i += 1

    # Check input parameters
    if size > 0 and len(input_dir) == 0:
        print("Input directory name is required with size")
        sys.exit(-1)

    if size > 0 and (test_data or len(check_data) > 0):
        print("Cannot train and test at the same time")
        sys.exit(-1)

    if test_data and len(check_data) > 0:
        print("You can either test or ask for recognition")
        sys.exit(-1)

    if (len(class1) > 0 or len(class2) > 0) and all_data:
        print("You can either run 1 vs 1 or 1 vs all")
        sys.exit(-1)

    if len(class1) == 0 and len(class2) == 0 and not all_data:
        print("Either --1vs1 or --all is mandatory")
        sys.exit(-1)

    if (len(class1) == 0 and len(class2) > 0) or (len(class1) > 0 and len(class2) == 0):
        print("Both class1 and class2 are mandatory")
        sys.exit(-1)

    if size > 0:
        train_set, test_set = split_sets(input_dir, "*.json", size)

        create_features(train_set, test_set, input_dir, fs_config, class1, class2)

        create_model(fs_config, class1, class2)

    if test_data:
        test_all(fs_config)

    if len(class1) > 0 and len(class2) > 0:
        test_one_vs_one(fs_config, class1, class2)

    if len(check_data) > 0:
        if not os.path.exists(check_data):
            print("Cannot access image feature {}".format(check_data))
            sys.exit(-1)
        else:
            check_image(fs_config, check_data)


if __name__ == "__main__":
    main()
