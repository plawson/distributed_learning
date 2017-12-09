import boto3
from pyspark import SparkContext
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
# from io import StringIO
from io import BytesIO
import numpy as np
import sys
import os
import uuid
import json
import shutil
import argparse

config = {
    "protocol": "s3n://",
    "bucket": "oc-plawson",
    "image_key": "distributed_learning/my_images",
    "sep": "/",
    "features_json_dir": "distributed_learning/json",
    "features_json_dir2": "distributed_learning/json2"
}

sc = SparkContext(appName="animals_identification")


def get_image_list():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(config['bucket'])
    image_list = []
    for obj in bucket.objects.filter(Prefix=config['image_key']):
        image_list.append(obj.key)

    return image_list


def create_features(local_dir):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(config['bucket'])
    os.mkdir(local_dir)
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    for obj in bucket.objects.filter(Prefix=config['image_key']):
        image_path = local_dir + config['sep'] + os.path.basename(obj.key)
        s3.Bucket(config['bucket']).download_file(obj.key, image_path)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        s3.Object(config['bucket'], config['features_json_dir'] + config['sep'] + os.path.basename(obj.key) + ".json")\
            .put(Body=json.dumps(features.tolist()[0]))

    shutil.rmtree(local_dir)


def create_feature(file):
    s3 = boto3.resource('s3')
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    local_file = './' + str(uuid.uuid4()) + os.path.basename(file)
    s3.Bucket(config['bucket']).download_file(file, local_file)
    img = image.load_img(local_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    s3.Object(config['bucket'], config['features_json_dir2'] + config['sep'] + os.path.basename(file) + ".json") \
        .put(Body=json.dumps(features.tolist()[0]))
    os.remove(local_file)


def is_s3_file_exists(bucket, file):
    client = boto3.client('s3')
    results = client.list_objects(Bucket=bucket, Prefix=file)
    return 'Contents' in results


def read_jpg_file(line):
    # extracting number of the file and the breedname
    filename = line[0]
    pos_last_slash = len(filename) - filename[::-1].find("/")
    filename = filename[pos_last_slash:]
    suffixe = ".jpg"
    len_suffixe = len(suffixe)
    pos_last_underscore = len(filename) - filename[::-1].find("_")
    numfile = int(filename[pos_last_underscore:-len_suffixe])
    breedname = filename[:(pos_last_underscore-1)]
    # extracting features from jpg #inspect.getsourcelines(image.load_img)
    target_size = (224, 224)
    img = image.pil_image.open(BytesIO(line[1]))  # img = np.asarray(PIL.Image.open(StringIO(raw_img)))
    img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # the model must be instantiate directly in the function
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    features = model.predict(x)
    features = np.asarray(features.tolist()[0])
    return numfile, breedname, features


def get_all_features(aconfig):
    binary_files_path = aconfig['protocol'] + aconfig['bucket'] + aconfig['sep'] + aconfig['image_key']
    rdd_all = sc.binaryFiles(binary_files_path) \
        .map(read_jpg_file).persist()
    # rdd_all = sc.binaryFiles(binary_files_path)
    return rdd_all


def do_1vs1(class_one, class_two, size, num_iter, aconfig):
    rdd_train = get_all_features(aconfig) \
            .filter(lambda data: data[0] <= size and (data[1] == class_one or data[1] == class_two)) \
            .map(lambda data: LabeledPoint((0.0 if data[1] == class_one else 1.0), data[2]))
    rdd_test = get_all_features(config) \
            .filter(lambda data: data[0] > size and (data[1] == class_one or data[1] == class_two)) \
            .map(lambda data: LabeledPoint((0.0 if data[1] == class_one else 1.0), data[2]))

    model = SVMWithSGD.train(rdd_train, num_iter)


def main():
    # Application configuration
    """
    app_config = {
        "protocol": "s3://",
        "bucket": "oc-plawson",
        "image_key": "distributed_learning/images",
        "sep": "/",
        "features_json_dir": "distributed_learning/json",
        "features_json_dir2": "distributed_learning/json2"
    }
    """

    parser = argparse.ArgumentParser()
    group_class = parser.add_mutually_exclusive_group(required=True)
    group_class.add_argument('--1vs1', help="Classification type, provide classX,classY")
    group_class.add_argument('--1vsAll', help="Classification type, provide classX")
    parser.add_argument('--size', required=True, help="Size of the training set", type=int)
    parser.add_argument('--iter', required=True, help="Number of iterations", type=int)

    args = parser.parse_args()

    size = args.size
    num_iter = args.iter
    classx_classy = vars(args)['1vs1']
    classall = vars(args)['1vsAll']

    class_one = None
    class_two = None
    if None is not classx_classy:
        if not len(classx_classy.split(',')) == 2:
            print("for --1vs1, please provide 2 classes separated by a comma (,) only, you provided: {}".
                  format(classx_classy))
            sys.exit(-1)
        if classx_classy.find(',') == -1 or classx_classy.find(',') == 0 or classx_classy.find(',') == \
                (len(classx_classy) - 1):
            print("for --1vs1, please provide 2 classes separated by a comma (,), you provided: {}".
                  format(classx_classy))
            sys.exit(-1)
        class_one = classx_classy[0:classx_classy.find(',')].strip()
        class_two = classx_classy[(classx_classy.find(',') + 1):].strip()
        do_1vs1(class_one, class_two, size, num_iter, config)

    # local_dir = str(uuid.uuid4())
    # create_features(app_config, local_dir)

    # sc.parallelize(get_image_list()).map(create_feature).collect()


if __name__ == "__main__":
    main()
