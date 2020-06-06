import cv2, grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import numpy as np
import tensorflow as tf
from inferencing import saved_model_config
from inferencing.saved_model_preprocess import ForwardModel
import requests
import json
import time
import datetime
import threading 
from matplotlib import pyplot
from mrcnn import visualize

results = []

host = saved_model_config.ADDRESS
PORT_GRPC = saved_model_config.PORT_NO_GRPC
RESTAPI_URL = saved_model_config.REST_API_URL

channel = grpc.insecure_channel(str(host) + ':' + str(PORT_GRPC), options=[('grpc.max_receive_message_length', saved_model_config.GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


request = predict_pb2.PredictRequest()
request.model_spec.name = saved_model_config.MODEL_NAME
request.model_spec.signature_name = saved_model_config.SIGNATURE_NAME

model_config = saved_model_config.MY_INFERENCE_CONFIG
preprocess_obj = ForwardModel(model_config)

print("config", model_config.IMAGE_RESIZE_MODE)

img2='/home/user/Desktop/bolt/poly/val/168 d.jpg'

def detect_mask_single_image_using_grpc(image):
    img2='/home/user/Desktop/bolt/poly/val/121 d.jpg'
    image2=cv2.imread(img2)
    images2=np.expand_dims(image2, axis=0)

    img3='/home/user/Desktop/bolt/poly/val/112 d.jpg'
    image3=cv2.imread(img3)
    images3=np.expand_dims(image3, axis=0)

    images1 = np.expand_dims(image, axis=0)

    images=[]
    images.append(images1[0])
    images.append(images2[0])

    images = np.expand_dims(image, axis=0)

    molded_images, image_metas, windows = preprocess_obj.mold_inputs(images)
    molded_images = molded_images.astype(np.float32)
    image_metas = image_metas.astype(np.float32)
    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
    print("Ani modi")
    # Anchors
    anchors = preprocess_obj.get_anchors(image_shape)
    # anchors = np.broadcast_to(anchors, (len(images),) + anchors.shape)

    anchors = np.broadcast_to(anchors, (len(images),) + anchors.shape)

    # print("shape",molded_images.shape)
    # print("shape",image_metas.shape)
    # print("shape",anchors.shape)

    request.inputs[saved_model_config.INPUT_IMAGE].CopyFrom(
        tf.contrib.util.make_tensor_proto(molded_images, shape=molded_images.shape))
    request.inputs[saved_model_config.INPUT_IMAGE_META].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_metas, shape=image_metas.shape))
    request.inputs[saved_model_config.INPUT_ANCHORS].CopyFrom(
        tf.contrib.util.make_tensor_proto(anchors, shape=anchors.shape))

    result = stub.Predict(request, 60.)
    print("result")
    result_dict = preprocess_obj.result_to_dict(images, molded_images, windows, result)[0]
    return result_dict


def detect_mask_single_image_using_restapi(image):
    img2='/home/user/Desktop/bolt/poly/val/168 d.jpg'
    image2=cv2.imread(img2)
    images2=np.expand_dims(image2, axis=0)

    images1 = np.expand_dims(image, axis=0)

    images=[]
    images.append(images1[0])
    images.append(images2[0])
    # print("images2 ",images1[0])
    images = np.expand_dims(image, axis=0)
    molded_images, image_metas, windows = preprocess_obj.mold_inputs(images)

    molded_images = molded_images.astype(np.float32)

    image_shape = molded_images[0].shape

    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    anchors = preprocess_obj.get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (len(images),) + anchors.shape)

    # print("anchors",len(anchors))

    # anchors = [anchors for i in range(2)]

    # print("anchors",anchors[0])

    # print("molded",molded_images[2])

    # response body format row wise.
    data = {'signature_name': saved_model_config.SIGNATURE_NAME,
            'instances': [{saved_model_config.INPUT_IMAGE: molded_images[0].tolist(),
                           saved_model_config.INPUT_IMAGE_META: image_metas[0].tolist(),
                           saved_model_config.INPUT_ANCHORS: anchors[0].tolist()}]}

    start=time.time()
    response = requests.post(RESTAPI_URL, data=json.dumps(data), headers={"content-type":"application/json"})
    result = json.loads(response.text)
    # print("ani",result)
    result = result['predictions'][0]
    end=time.time()
    # print("result ",result['detection'])
    print("TIME: ", end-start)
    result_dict = preprocess_obj.result_to_dict(images, molded_images, windows, result, is_restapi=True)[0]

    results.append(result_dict)
    return result_dict


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to Image', required=True)
    parser.add_argument('-t', '--type', help='Type of call [restapi, grpc]', default='restapi')
    args = vars(parser.parse_args())
    image_path = args['path']
    call_type = args['type']

    if not os.path.exists(image_path):
        print(image_path, " -- Does not exist")
        exit()

    # img2='/home/user/Desktop/bolt/poly/val/168 d.jpg'
    # image2=cv2.imread(img2)

    images = []
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path,filename))
        if img is not None:
            images.append(img)

    # image = cv2.imread(image_path)
    # if image is None:
    #     print("Image path is not proper")
    #     exit()

    start=time.time()
    threads = [None] * len(images)

    if call_type == 'restapi':
        i=0
        for image in images:
            threads[i] = threading.Thread(target=detect_mask_single_image_using_restapi, args=(image,)) 
            # t2 = threading.Thread(target=detect_mask_single_image_using_restapi, args=(image2,)) 
            threads[i].start() 
            i=i+1
        
        for i in range(len(threads)):
            threads[i].join()

    else:
        result = detect_mask_single_image_using_grpc(image)

    print("*" * 60)
    print("RESULTS:")
    print(results)   
    print("*" * 60)

    # print("*" * 60)
    # print("RESULTS:")
    # print(results[1])
    # print("*" * 60)

    end=time.time()

    print("Total Time:", end-start)
    print("Mean Time:", (end-start)/len(images))

    visualize.display_instances(images[0], results[0]['rois'], results[0]['mask'], results[0]['class'], 
                            ['screw','screw','bolt'], results[0]['scores'],
                            title="Predictions")