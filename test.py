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
    i = 0
    for file in os.listdir(local_image_dir):
        if i > 0:
            break
        image_path = local_image_dir + config['sep'] + file
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        print(json.dumps(features.tolist()[0]))
        s3.Object(config['bucket'], config['features_json_dir'] + config['sep'] + file + ".json")\
            .put(Body=json.dumps(features.tolist()[0]))
        i += 1


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

    # image.load_img('s3://oc-plawson/distributed_learning/images/yorkshire_terrier_86.jpg', target_size=(224, 224))

    # local_image_dir = './' + str(uuid.uuid4())
    local_image_dir = './d7f2680c-270b-44cd-bed5-a2855c4ad56d'
    # get_image_files(app_config, local_image_dir)
    create_features(app_config, local_image_dir)

    # rdd = sc.parallelize(get_image_list(app_config))


if __name__ == "__main__":
    main()
