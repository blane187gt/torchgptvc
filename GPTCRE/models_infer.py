import numpy as np

def infer_pitch(model, spectrogram):
    predictions = model.predict(spectrogram)
    pitch = np.argmax(predictions, axis=1)
    return pitch
