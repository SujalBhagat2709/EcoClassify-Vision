#OpenCV File for AI Waste Irrigation model
import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/waste_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels (in same order as your folders)
labels = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']

# Setup webcam
cap = cv2.VideoCapture(0)  # Use 1 or 2 if default cam doesn't work

IMG_SIZE = 224  # Input size of the model

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

    # Make prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    confidence = np.max(output)    #-> new one
    prediction = np.argmax(output)
    label = labels[prediction]

    # Only display if confidence > 0.75   -> new one
    if confidence > 0.75:
        display_text = f'{label} ({confidence:.2f})'
    
    else:
        display_text = "No confident prediction"
    
    # Display result
    cv2.putText(frame, f'Prediction: {label}', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
