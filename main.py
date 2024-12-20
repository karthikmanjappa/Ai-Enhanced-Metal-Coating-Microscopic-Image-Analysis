import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from io import BytesIO

# Class Definitions
class ImageProcessor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    @staticmethod
    def apply_clahe(image, clip_limit=10, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    @staticmethod
    def split_image(image, rows=2, cols=2):
        h, w = image.shape[:2]
        row_h = h // rows
        col_w = w // cols
        return [
            image[r * row_h:(r + 1) * row_h, c * col_w:(c + 1) * col_w]
            for r in range(rows)
            for c in range(cols)
        ]

    def segment(self, image_part):
        img = cv2.resize(image_part, (256, 256))
        input_image = np.stack((img,) * 3, axis=-1) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        prediction = (self.model.predict(input_image)[0, :, :, 0] > 0.6).astype(np.uint8)
        return prediction

    @staticmethod
    def merge_parts(parts, original_shape, rows=2, cols=2):
        h, w = original_shape[:2]
        row_h = h // rows
        col_w = w // cols
        merged_image = np.zeros(original_shape, dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                resized_part = cv2.resize(parts[r * cols + c], (col_w, row_h))
                merged_image[r * row_h:(r + 1) * row_h, c * col_w:(c + 1) * col_w] = resized_part

        return merged_image

    def process_image(self, image):
        parts = self.split_image(image)
        clahe_parts = [self.apply_clahe(part) for part in parts]
        segmented_parts = [self.segment(part) for part in clahe_parts]
        segmented_image = self.merge_parts(segmented_parts, image.shape)
        return segmented_image

    def apply_contour(self, segmented_image):
        ret, binary = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        highlighted_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cont_img = cv2.drawContours(highlighted_img, contours, -1, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        return cont_img, contours

class Parameters:
    @staticmethod
    def melted_and_unmelted_fraction(segmented_image):
        count_melted = np.sum(segmented_image == 0)
        count_unmelted = np.sum(segmented_image == 1)
        total_pixels = segmented_image.size
        return {
            "Fraction Melted (%)": (count_melted / total_pixels) * 100,
            "Fraction Unmelted (%)": (count_unmelted / total_pixels) * 100,
        }

    @staticmethod
    def area(contours):
        contour_data = []
        for i, cont in enumerate(contours):
            x, y, width, height = cv2.boundingRect(cont)
            aspect_ratio = width / float(height) if height != 0 else 0
            area = cv2.contourArea(cont)
            contour_data.append({
                "Contour": i + 1,
                "Width": width,
                "Height": height,
                "Aspect Ratio": aspect_ratio,
                "Area": area,
            })
        return pd.DataFrame(contour_data)



# Streamlit Interface
st.title("AI Based Metal Coating Image Analysis Software")
st.write("Upload an image to Analyze and view results.")

# Load model 
model_path = r"C:\Users\karth\OneDrive\Desktop\Internship\unet_model2 (1).h5"  # Update with your model's path
processor = ImageProcessor(model_path)

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "tif"])
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    #st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image
    segmented_image = processor.process_image(image)
    cont_img, contours = processor.apply_contour(segmented_image)

    # Calculate parameters
    params = Parameters.melted_and_unmelted_fraction(segmented_image)
    df_contours = Parameters.area(contours)

    # Display results
    st.subheader("Processed Images")

    # Horizontal image display
    col1, col2, col3 = st.columns([5, 5, 5])
    with col1:
        st.image(image, caption="Original Image",width=500,use_container_width=True)
    with col2:
        segmented_image = (segmented_image * 255).astype(np.uint8)
        st.image(segmented_image, caption="Segmented Mask", use_container_width=True,width=500)
    with col3:
        st.image(cont_img, caption="Contour Image", use_container_width=True,width=500)

    st.subheader("Parameters")
    st.write(params)
    st.write(df_contours)

    # Download DataFrame as Excel
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()

    excel_data = convert_df_to_excel(df_contours)
    st.download_button(
        label="Download Data as Excel",
        data=excel_data,
        file_name="contour_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
