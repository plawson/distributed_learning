import boto3
from pyspark import SparkContext
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import uuid
import json
import shutil

config = {
    "protocol": "s3://",
    "bucket": "oc-plawson",
    "image_key": "distributed_learning/images",
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

    # local_dir = str(uuid.uuid4())
    # create_features(app_config, local_dir)

    sc.parallelize(get_image_list()).map(create_feature).collect()


if __name__ == "__main__":
    main()
