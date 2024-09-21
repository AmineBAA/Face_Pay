import cv2
import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace

# Function to extract face and analyze using DeepFace
def analyze_face(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        return result
    except Exception as e:
        st.error(f"Error in face analysis: {str(e)}")
        return None

# Load reference image and get its face embedding using DeepFace
def load_reference_image(uploaded_image):
    if uploaded_image is not None:
        # Convert the uploaded image (PIL) to an OpenCV format
        pil_image = Image.open(uploaded_image)
        open_cv_image = np.array(pil_image)
        
        try:
            ref_face = DeepFace.represent(open_cv_image, enforce_detection=False)
            if ref_face:
                return ref_face[0]["embedding"]  # Extract embedding
            else:
                st.error("No face detected in the reference image.")
                return None
        except Exception as e:
            st.error(f"Error in face detection: {str(e)}")
            return None
    else:
        st.error("Please upload a reference image.")
        return None

# Compare face embeddings using cosine similarity
def compare_faces(embedding1, embedding2, threshold=0.3):
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))  # Euclidean distance
    return distance < threshold  # Return True if distance is less than the threshold

# Streamlit interface
def main():
    st.title("Enhanced Face Recognition with Code Matching using DeepFace")

    # Upload a reference image
    uploaded_reference
