import boto3
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from io import BytesIO
import numpy as np
import json

config = {
    "protocol": "s3n://",
    "bucket": "oc-plawson",
    "image_key": "distributed_learning/test_images",
    "sep": "/",
    "features_key": "distributed_learning/json",
    "features_json_dir2": "distributed_learning/json2"
}


s3 = boto3.resource('s3')
base_model = VGG16(weights='imagenet')
bucket = s3.Bucket(config['bucket'])
for obj in bucket.objects.filter(Prefix=config['image_key']):
    target_size = (224, 224)
    img = image.pil_image.open(BytesIO(obj.get()['Body'].read()))
    img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    features = model.predict(x)
    # features = np.asarray(features.tolist()[0])
    s3.Object(config['bucket'], config['features_json_dir2'] + config['sep'] + obj.key[(len(obj.key) - obj.key[::-1].find("/")):] + ".json")\
        .put(Body=json.dumps(features.tolist()[0]))

