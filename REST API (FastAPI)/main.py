from fastapi import FastAPI, File, UploadFile
#import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


MODEL = tf.keras.models.load_model('model.keras', compile=False)


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/")
async def root():
    return "Successfully Connected"

@app.get("/ping")
async def ping():
    return "Hi, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...) ):
    image = read_file_as_image(await file.read())
    a = tf.image.resize(image, [256,256])
    img_batch = np.expand_dims(a, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    } 

PORT = int(os.get('PORT', 8000))
HOST = '0.0.0.0'

if __name__ == "__main__":
       #uvicorn.run(app, host='localhost', port=9560)
       #uvicorn.run(app)
       uvicorn.run('app:app', host = HOST, port = PORT, reload = True)
       
 
    