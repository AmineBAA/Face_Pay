import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load OpenCV DNN Face Detector
def load_dnn_face_detector():
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

# Detect faces using DNN
def detect_faces_dnn(net, image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            faces.append(face)
    return faces

# Load reference image and get its face
def load_reference_face(uploaded_image, net):
    if uploaded_image is not None:
        # Convert the uploaded image (PIL) to an OpenCV format
        pil_image = Image.open(uploaded_image)
        open_cv_image = np.array(pil_image)
        
        # Detect face in the reference image
        faces = detect_faces_dnn(net, open_cv_image)
        if faces:
            return faces[0]  # Return the first detected face
        else:
            st.error("No face detected in the reference image.")
            return None
    else:
        st.error("Please upload a reference image.")
        return None

# Compare faces using Histogram Matching (alternative to face embeddings)
def compare_faces(face1, face2):
    # Convert faces to grayscale for comparison
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    
    # Resize face2 to match face1 size
    face2_resized = cv2.resize(face2_gray, (face1_gray.shape[1], face1_gray.shape[0]))
    
    # Compute histogram for both faces
    hist1 = cv2.calcHist([face1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2_resized], [0], None, [256], [0, 256])
    
    # Normalize histograms and compare
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Use correlation or any other similarity metric (we're using correlation here)
    comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return comparison > 0.7  # Adjust threshold as needed

# Streamlit interface
def main():
    st.title("Face Recognition System with Code Matching using OpenCV DNN")

    # Load DNN model
    net = load_dnn_face_detector()

    # Upload a reference image
    uploaded_reference_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
    reference_face = load_reference_face(uploaded_reference_image, net)
    
    if reference_face is None:
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
        
        # Detect face in the captured frame
        faces_in_frame = detect_faces_dnn(net, frame)
        
        if faces_in_frame:
            face_in_frame = faces_in_frame[0]
            
            # Compare with the reference face using histogram comparison
            face_match = compare_faces(reference_face, face_in_frame)

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
