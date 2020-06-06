# Your Inference Config Class
# Replace your own config
# MY_INFERENCE_CONFIG = YOUR_CONFIG_CLASS
import coco
import final

config = final.CustomConfig()
class InferenceConfig(config.__class__):
# Run detection on one image at a time
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.9
config = InferenceConfig()
config.display()

MY_INFERENCE_CONFIG = config

# Tensorflow Model server variable
ADDRESS = 'localhost'
PORT_NO_GRPC = 5001
PORT_NO_RESTAPI = 5000
MODEL_NAME = 'mask'
REST_API_URL = "http://%s:%s/v1/models/%s:predict" % (ADDRESS, PORT_NO_RESTAPI, MODEL_NAME)


# TF variable name
OUTPUT_DETECTION = 'mrcnn_detection/Reshape_1'
OUTPUT_CLASS = 'mrcnn_class/Reshape_1'
OUTPUT_BBOX = 'mrcnn_bbox/Reshape'
OUTPUT_MASK = 'mrcnn_mask/Reshape_1'
INPUT_IMAGE = 'input_image'
INPUT_IMAGE_META = 'input_image_meta'
INPUT_ANCHORS = 'input_anchors'
OUTPUT_NAME = 'predict_images'


# Signature name
SIGNATURE_NAME = 'serving_default'

# GRPC config
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3 # Max LENGTH the GRPC should handle
