import streamlit as st
from PIL import Image
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load your Detectron 2 model configuration and weights
cfg = get_cfg()
cfg.merge_from_file(r"C:/Users/preethi sirimalla/Downloads/maskrcnn/semantic-segmentation/mask_rcnn_R_101_FPN_3x")
cfg.MODEL.WEIGHTS = r"C:/Users/preethi sirimalla/Downloads/maskrcnn/semantic-segmentation/mask_rcnn_R_101_FPN_3x/2024-01-17-11-33-26/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Adjust threshold as needed
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# Function to calculate IoU between two masks
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Function to check contamination based on IoU threshold
def is_contaminated(predictions_class1, predictions_class2, iou_threshold=0.05):
    masks_class1 = predictions_class1["instances"].pred_masks.cpu().numpy()
    masks_class2 = predictions_class2["instances"].pred_masks.cpu().numpy()

    for mask_class1 in masks_class1:
        for mask_class2 in masks_class2:
            iou = calculate_iou(mask_class1, mask_class2)
            if iou > iou_threshold:
                return True  # Image is contaminated

    return False  # Image is not contaminated

# Streamlit app
def main():
    st.title("Contamination Detection App")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_array = np.array(image)
        height, width, channels = image_array.shape
        
        # Make predictions for both classes
        predictions_class1 = predictor(image_array)
        # Replace "path/to/your/image2.jpg" with the path to another image for class 2 predictions
        #predictions_class2 = predictor(Image.open("path/to/your/image2.jpg"))
        predictions_class2 = predictor(image_array)

        print(predictions_class1)

        # Check if the image is contaminated based on IoU threshold
        contaminated = is_contaminated(predictions_class1, predictions_class2)

        # Display results
        st.write("Is Contaminated:", contaminated)

        # Optionally, visualize the predictions
        v = Visualizer(np.array(image), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(predictions_class1["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], caption="Class 1 Predictions", use_column_width=True)

        v = Visualizer(np.array(image), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(predictions_class2["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], caption="Class 2 Predictions", use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
