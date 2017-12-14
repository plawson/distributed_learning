import boto3
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# from io import StringIO
from io import BytesIO
import numpy as np
import sys
import os
import uuid
import json
import shutil
import argparse


# sc = SparkContext(appName="animals_identification")


def create_features(config):
    s3 = boto3.resource('s3')
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    bucket = s3.Bucket(config['bucket'])

    for obj in bucket.objects.filter(Prefix=config['image_key']):  # Read all jpeg files for the bucket key
        img = image.load_img(BytesIO(obj.get()['Body'].read()), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        # Save the features to a .txt file
        pos_last_slash = len(obj.key) - obj.key[::-1].find("/")
        filename = obj.key[pos_last_slash:]
        suffix = ".jpg"
        len_suffix = len(suffix)
        pos_last_underscore = len(filename) - filename[::-1].find("_")
        num_file = int(filename[pos_last_underscore:-len_suffix])
        breed_name = filename[:(pos_last_underscore - 1)]
        s3.Object(config['bucket'], config['features_key'] + config['sep']
                  + obj.key[(len(obj.key) - obj.key[::-1].find("/")):] + ".txt") \
            .put(Body=breed_name + ',' + str(num_file) + ',' + json.dumps(features.tolist()[0])[1:][:-1])


def s3_object_exists(bucket, file):
    client = boto3.client('s3')
    results = client.list_objects(Bucket=bucket, Prefix=file)
    return 'Contents' in results


def main():
    config = {
        "protocol": "s3://",
        "bucket": "oc-plawson",
        "image_key": "distributed_learning/test_images",
        "sep": "/",
        "features_key": "distributed_learning/test_features"
    }

    # Define command line parameters
    parser = argparse.ArgumentParser()
    group_class = parser.add_mutually_exclusive_group(required=True)
    group_class.add_argument('--1vs1', help="Classification type, provide classX,classY")
    group_class.add_argument('--1vsAll', help="Classification type, provide classX")
    parser.add_argument('--size', required=True, help="Size of the training set", type=int)
    parser.add_argument('--iter', required=True, help="Number of iterations", type=int)

    # Parse and check command line arguments
    print('Parsing command line parameters...')
    args = parser.parse_args()

    # Register input parameters' value
    size = args.size
    num_iter = args.iter
    classx_classy = vars(args)['1vs1']
    class_all = vars(args)['1vsAll']

    # Process 1vs1 command line parameters
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

    if not s3_object_exists(config['bucket'], config['features_key']):
        print('Creating all features...')
        create_features(config)


if __name__ == "__main__":
    main()
