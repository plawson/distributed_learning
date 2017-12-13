import boto3
from pyspark import SparkContext
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import uuid
import json

# sc = SparkContext(appName="animals_identification")


def get_image_files(config, local_dir):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(config['bucket'])
    os.mkdir(local_dir)

    for obj in bucket.objects.filter(Prefix=config['image_key']):
        s3.Bucket(config['bucket']).download_file(obj.key, local_dir + config['sep'] + os.path.basename(obj.key))


def is_s3_file_exists(bucket, file):
    client = boto3.client('s3')
    results = client.list_objects(Bucket=bucket, Prefix=file)
    return 'Contents' in results


def create_keras_model(keras_model_local_file, bucket, keras_model_s3_file):
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    model.save(keras_model_local_file)
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(keras_model_local_file, bucket, keras_model_s3_file)
    os.remove(keras_model_local_file)


def load_keras_model(config):
    file = './' + str(uuid.uuid4()) + config['keras_model_file']
    s3 = boto3.resource('s3')
    s3.Bucket(config['bucket']).download_file(config['keras_model'], file)
    keras_model = load_model(file, compile=True)
    os.remove(file)
    return keras_model


def create_features(config, local_image_dir):
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    s3 = boto3.resource('s3')
    for file in os.listdir(local_image_dir):
        image_path = local_image_dir + config['sep'] + file
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        print(json.dumps(features.tolist()[0]))
        s3.Object(config['bucket'], config['features_json_dir'] + config['sep'] + file + ".json")\
            .put(Body=json.dumps(features.tolist()[0]))


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


def read_jpg_file(line):
    # extracting number of the file and the breedname
    filename = line[0]
    pos_last_slash = len(filename) - filename[::-1].find("/")
    filename = filename[pos_last_slash:]
    suffixe = ".jpg"
    len_suffixe = len(suffixe)
    pos_last_underscore = len(filename) - filename[::-1].find("_")
    numfile = int(filename[pos_last_underscore:-len_suffixe])
    breedname = filename[:(pos_last_underscore - 1)]
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
    rdd_all = sc.binaryFiles(binary_files_path, minPartitions=8) \
        .map(read_jpg_file).persist()
    # rdd_all = sc.binaryFiles(binary_files_path)
    return rdd_all


def main():
    # Application configuration
    app_config = {
        "protocol": "s3://",
        "bucket": "oc-plawson",
        "image_key": "distributed_learning/images",
        "sep": "/",
        "keras_model": "keras_model/model.h5",
        "keras_local_file": "./model.h5",
        "keras_model_file": "model.h5",
        "features_json_dir": "distributed_learning/json"
    }

    #if not is_s3_file_exists(app_config['bucket'], app_config['keras_model']):
        #create_keras_model(app_config['keras_local_file'], app_config['bucket'], app_config['keras_model'])

    #keras_model = load_keras_model(app_config)

    local_image_dir = './' + str(uuid.uuid4())
    get_image_files(app_config, local_image_dir)
    create_features(app_config, local_image_dir)

    # rdd = sc.parallelize(get_image_list(app_config))


if __name__ == "__main__":
    main()
