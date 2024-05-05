import cv2
from PIL import Image
import numpy as np
from utils.utils import load_best_model_and_transform, get_transform_img


def setup_camera():
    cap = cv2.VideoCapture(0)
    resolution_width, resolution_height = 2160, 3840
    cap.set(3, resolution_width)  # Set the width
    cap.set(4, resolution_height)  # Set the height
    return cap


def setup_window():
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', 1920, 1080)


def overlay_reduced_image(frame, reduced_image_np):
    reduced_height, reduced_width, _ = reduced_image_np.shape
    frame[0:reduced_height, 0:reduced_width] = reduced_image_np


def preprocess_frame(frame, transform, transform_img, model):
    result_frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    reduced_image = transform_img(result_frame_pil)
    reduced_image_np = cv2.cvtColor(np.array(reduced_image), cv2.COLOR_RGB2BGR)
    overlay_reduced_image(frame, reduced_image_np)

    image = transform(result_frame_pil).unsqueeze(0)  # Apply the transform and add a batch dimension
    prediction = model.predict_proba(image)
    return prediction


def display_result(frame, predicted_class, probability):
    if probability > 0.40:
        label = f"Resistor of {predicted_class} (Probability: {round(probability, 2)})"
    else:
        label = "not recognized"

    cv2.putText(frame, label, (10, 800), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 8)



def run():
    model, transform = load_best_model_and_transform()
    transform_img = get_transform_img()

    cap = setup_camera()
    setup_window()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        prediction = preprocess_frame(frame, transform, transform_img, model)
        predicted_class, probability = prediction['class'], prediction['prob']
        
        display_result(frame, predicted_class, probability)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
