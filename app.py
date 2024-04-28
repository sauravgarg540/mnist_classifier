import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from torch.nn import functional as F

from config import CHECKPOINT_DIR, IMAGE_SIZE, MODEL_NAME
from src.data.transforms import Normalize, ToTorchFormatTensor
from src.model.mnist_model import MNISTModel
from src.utils.utils import get_device

app = FastAPI()

# model for inference
device = get_device()
model = MNISTModel()
checkpoint_file = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
model.load_state_dict(torch.load(checkpoint_file)['model_state_dict'])
model.eval()
model.to(device)


def read_file_as_image(data: bytes) -> torch.Tensor:
    """
    Read image data from bytes, apply transforms and prepare it for inference.

    Args:
        data: Raw image data.

    Returns:
        Preprocessed image tensor.
    """

    data_transform = T.Compose([ToTorchFormatTensor(), Normalize()])

    # Convert bytes to numpy array
    nparr = np.frombuffer(data, np.uint8)

    # Read image in grayscale from numpy array and resize for inference
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)

    # Add batch dimension and apply transforms
    img = np.expand_dims(img, 0)
    img = data_transform(img)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class and confidence of a handwritten digit image.

    Args:
        file: Uploaded image file.

    Returns:
        class: predicted class
        confidence: confidence of prediction in percentage
    """

    try:  # A try block to handle any errors that may occur
        image = read_file_as_image(await file.read())  # Read the image file
        # Add an extra dimension to the image so that it matches the input shape of the model

        predictions = model(image.to(device))
        predictions = F.softmax(predictions)
        if 'cuda' in device:
            predictions = predictions.detach().cpu().numpy()[0]
        predicted_class = np.argmax(predictions)  # Get the predicted class
        confidence = np.max(predictions)  # Get the confidence of the prediction

        return {  # Return the prediction
            'class': predicted_class.item(),
            'confidence': confidence.item() * 100,
        }
    except Exception as e:  # If an error occurs
        raise HTTPException(
            status_code=400, detail=str(e)
        )  # Raise an HTTPException with the error message


if __name__ == "__main__":  # If the script is run directly
    uvicorn.run(app, host="localhost", port=8002)  # Run the FastAPI app using uvicorn
