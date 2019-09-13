import os
import sys

# Try to mute and then load Tensorflow and Keras
# Muting seems to not work lately on Linux in any way
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.client import device_lib
sys.stdin = stdin
sys.stderr = stderr

# Get list of all devices
devices = device_lib.list_local_devices()

# Print GPUs only
print('\n\n\nList of found GPUs:')
for device in devices:
    if device.device_type == 'GPU':
        print(device.physical_device_desc)
