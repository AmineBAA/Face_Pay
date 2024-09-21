import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to detect and crop face from an image
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Assuming we take the first detected face
        face_crop = gray_image[y:y + h, x:x + w]
        return face_crop
    else:
        return None

# Load reference image
def load_reference_image():
    ref_img = cv2.imread("My Photo.jpg")
    ref_face = detect_face(ref_img)
    if ref_face is not None:
        return ref_face
    else:
        st.error("No face detected in the reference image.")
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
    reference_image = load_reference_image()
    
    if reference_image is None:
        st.stop()  # Stop if no reference face is detected

    # User input for code matching
    user_code = st.text_input("Enter your code (e.g., '1234'):")
    correct_code = "1234"  # Set the correct code

    # Initialize webcam feed
    video_capture = cv2.VideoCapture(0)

    # Streamlit webcam display
    stframe = st.empty()  # For displaying video frame

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if not ret:
            st.error("Failed to capture video.")
            break

        # Detect face in the current video frame
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
            st.warning("No face detected in the current frame.")

        # Show the video frame on Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # End loop if user presses "q"
        if st.button("Stop"):
            break

    # Release the capture when done
    video_capture.release()

if __name__ == "__main__":
    main()
