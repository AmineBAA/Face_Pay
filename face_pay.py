import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to detect and crop face from an image
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the color image to grayscale (for face detection)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Assuming we take the first detected face
        face_crop = gray_image[y:y + h, x:x + w]
        return face_crop
    else:
        return None

# Load reference image
def load_reference_image(uploaded_image):
    if uploaded_image is not None:
        # Convert the uploaded image (PIL) to an OpenCV format
        pil_image = Image.open(uploaded_image)
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        
        ref_face = detect_face(open_cv_image)
        if ref_face is not None:
            return ref_face
        else:
            st.error("No face detected in the reference image.")
            return None
    else:
        st.error("Please upload a reference image.")
        return None

# Compare two faces
def compare_faces(face1, face2):
    face1_resized = cv2.resize(face1, (face2.shape[1], face2.shape[0]))
    difference = cv2.absdiff(face1_resized, face2)
    result = np.sum(difference)  # Sum of absolute differences
    return result

# Streamlit interface
def main():
    st.title("Face Recognition System with Code Matching")

    # Upload a reference image
    uploaded_reference_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
    reference_image = load_reference_image(uploaded_reference_image)
    
    if reference_image is None:
        st.stop()  # Stop if no reference face is detected

    # User input for code matching
    user_code = st.text_input("Enter your code (e.g., '1234'):")
    correct_code = "1234"  # Set the correct code

    # Use Streamlit's camera input for capturing an image from webcam
    camera_image = st.camera_input("Capture an image using webcam")

    if camera_image is not None:
        # Convert the captured camera image to OpenCV format
        img = Image.open(camera_image)
        frame = np.array(img)
        frame = frame[:, :, ::-1]  # Convert RGB to BGR for OpenCV

        # Detect face in the captured frame
        face_in_frame = detect_face(frame)
        
        if face_in_frame is not None:
            # Compare with the reference face
            comparison_result = compare_faces(reference_image, face_in_frame)

            # Define a matching threshold
            threshold = 100000  # Adjust this threshold as necessary

            if comparison_result < threshold:
                # If faces match
                if user_code == correct_code:
                    st.success(f"Face Matched! Welcome user with code {user_code}")
                else:
                    st.warning("Face matched, but code does not match.")
            else:
                st.warning("Face does not match.")
        else:
            st.warning("No face detected in the captured image.")

if __name__ == "__main__":
    main()
