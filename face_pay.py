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
        try:
            current_face_rep = DeepFace.represent(frame, enforce_detection=False)
            current_face_embedding = current_face_rep[0]["embedding"] if current_face_rep else None
        except Exception as e:
            st.error(f"Error in face representation: {str(e)}")
            current_face_embedding = None

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
