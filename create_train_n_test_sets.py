import glob
import sys
import os
import re
import json
import shutil
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext(appName="ImageRecognition")


def parsePoint(line):
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


def create_features(train_set, test_set, input_dir):
    if os.path.exists('./features/train_features.txt'):
        os.remove('./features/train_features.txt')
    if os.path.exists('./features/test_features.txt'):
        os.remove('./features/test_features.txt')

    for file_name in train_set['yorkshire_terrier']:
        with open(input_dir + '/' + file_name, 'r') as d:
            data = json.load(d)
            data = '0,' + ",".join(map(str, data))
            with open('./features/train_features.txt', 'a') as f:
                print(data, file=f)

    for file_name in train_set['wheaten_terrier']:
        with open(input_dir + '/' + file_name, 'r') as d:
            data = json.load(d)
            data = '1,' + ",".join(map(str, data))
            with open('./features/train_features.txt', 'a') as f:
                print(data, file=f)

    for file_name in test_set['yorkshire_terrier']:
        with open(input_dir + '/' + file_name, 'r') as d:
            data = json.load(d)
            data = '0,' + ",".join(map(str, data))
            with open('./features/test_features.txt', 'a') as f:
                print(data, file=f)

    for file_name in test_set['wheaten_terrier']:
        with open(input_dir + '/' + file_name, 'r') as d:
            data = json.load(d)
            data = '1,' + ",".join(map(str, data))
            with open('./features/test_features.txt', 'a') as f:
                print(data, file=f)


def create_model():
    # Load training data
    data = sc.textFile('./features/train_features.txt')
    parsed_data = data.map(parsePoint)

    # Build the model
    model = SVMWithSGD.train(parsed_data, iterations=100)

    # Evaluate the model on training data
    labels_and_preds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
    train_err = labels_and_preds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsed_data.count())
    print("Training Error = " + str(train_err))

    if os.path.exists('./model/ImageRecognitionModel') and os.path.isdir('./model/ImageRecognitionModel'):
        shutil.rmtree('./model/ImageRecognitionModel')

    # Save the model
    model.save(sc, './model/ImageRecognitionModel')


def usage(prog):
    print("Usage: " + prog + " --dir <input_directory> --size <train_size_by_class>")


def main():
    i = 0
    args = sys.argv[1:]
    if len(args) != 4 and len(args) != 1:
        usage(sys.argv[0])
        sys.exit(-1)

    input_dir = ""
    size = 0
    test_data = False
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
                print("Train size by class ({}) must be a number".format(args[i]))
                sys.exit(-1)
            else:
                size = int(args[i])
        elif args[i] == "--test":
            test_data = True

        i += 1

    if size > 0:

        train_set, test_set = split_sets(input_dir, "*.json", size)

        create_features(train_set, test_set, input_dir)

        create_model()

    if test_data:
        # Load test data
        data = sc.textFile('./features/test_features.txt')
        parsed_data = data.map(parsePoint)
        # Load the model
        model = SVMModel.load(sc, './model/ImageRecognitionModel')
        # Test the model on test data
        labels_and_preds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
        test_err = labels_and_preds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsed_data.count())
        print("Test Error = " + str(test_err * 100) + "%")


if __name__ == "__main__":
    main()
