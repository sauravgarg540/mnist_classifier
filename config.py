import os

# set paths for mnist_to_imagenet.py, train.py, eval.py
UNPROCESSED_DATA_ROOT_DIR = "E:\\Datasets\\mnist"
DATA_ROOT_DIR = "E:\\Datasets\\processed_mnist"

# Setting for training, eval and model serving
# set paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# Training settings
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
CRITERION = "cross_entropy"
IMAGE_SIZE = (28, 28)


NUM_CLASSES = 10
MODEL_NAME = "model.pt"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
