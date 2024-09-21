import cv2
import numpy as np
import streamlit as st
from PIL import Image
import face_recognition

# Function to extract face embeddings using face_recognition
def extract_face_embedding(image):
    # Convert the color image (BGR) to RGB
    rgb_image = image[:, :, ::-1]
    
    # Find face locations
    face_locations = face_recognition.face_locations(rgb_image)
    
    # Get face embeddings for the faces in the image
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        return face_encodings[0]  # Assuming we take the first face found
    else:
        return None

# Load reference image and get its face embedding
def load_reference_image(uploaded_image):
    if uploaded_image is not None:
        # Convert the uploaded image (PIL) to an OpenCV format
        pil_image = Image.open(uploaded_image)
        open_cv_image = np.array(pil_image)
        
        # Get face embedding from the reference image
        ref_face_embedding = extract_face_embedding(open_cv_image)
        
        if ref_face_embedding is not None:
            return ref_face_embedding
        else:
            st.error("No face detected in the reference image.")
            return None
    else:
        st.error("Please upload a reference image.")
        return None

# Compare face embeddings using Euclidean distance
def compare_faces(embedding1, embedding2, threshold=0.6):
    distance = np.linalg.norm(embedding1 - embedding2)  # Euclidean distance
    return distance < threshold  # Return True if distance is less than the threshold

# Streamlit interface
def main():
    st.title("Enhanced Face Recognition with Code Matching")

    # Upload a reference image
    uploaded_reference_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
    reference_embedding = load_reference_image(uploaded_reference_image)
    
    if reference_embedding is None:
        st.stop()  # Stop if no reference face embedding is detected

    # User input for code matching
    user_code = st.text_input("Enter your code (e.g., '1234'):")
    correct_code = "1234"  # Set the correct code

    # Use Streamlit's camera input for capturing an image from webcam
    camera_image = st.camera_input("Capture an image using webcam")

    if camera_image is not None:
        # Convert the captured camera image to OpenCV format
        img = Image.open(camera_image)
        frame = np.array(img)
        
        # Get face embedding from the captured frame
        current_face_embedding = extract_face_embedding(frame)
        
        if current_face_embedding is not None:
            # Compare with the reference face embedding
            face_match = compare_faces(reference_embedding, current_face_embedding)

            if face_match:
                if user_code == correct_code:
                    st.success(f"Face and code matched! Welcome, user with code {user_code}.")
                else:
                    st.warning("Face matched, but code does not match.")
            else:
                st.warning("Face does not match.")
        else:
            st.warning("No face detected in the captured image.")

if __name__ == "__main__":
    main()
