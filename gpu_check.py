import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('GPU is available')
    # Set the GPU as the default device
    tf.config.experimental.set_memory_growth(gpus[0], True)

    print(len(gpus))
else:
    print('GPU is not available')
