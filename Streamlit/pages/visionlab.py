import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
import torch
import torch.nn.functional as F
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from torchvision.models import Inception_V3_Weights, ResNet50_Weights
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import make_gradcam_heatmap, make_gradcam_heatmap_keras, get_canny_edge

# --- App Title ---
st.title("Vision Lab")
class_names = ['BAS', 'EOS', 'EBO', 'IG', 'LYT', 'MON', 'NGS', 'PLA']

@st.cache_resource
# --- Load Models ---
def load_inception():
    return torch.load('./Streamlit/inceptionv3.pth', weights_only=False,map_location=torch.device('cpu'))
def load_resnet():
    return torch.load('./Streamlit/resnet_model.pth', weights_only=False,map_location=torch.device('cpu'))
def load_vgg16():
    return load_model('./Streamlit/vgg16_model.h5')

# --- Model Selection ---
selected_model_name = st.selectbox("Choose a model", ["None", "InceptionV3", "ResNet50", "VGG16"])
selected_model = None
if selected_model_name != "None":
    with st.spinner(f"Loading {selected_model_name}..."):
        if selected_model_name == "InceptionV3":
            selected_model = load_inception()
        elif selected_model_name == "ResNet50":
            selected_model = load_resnet()
        elif selected_model_name == "VGG16":
            selected_model = load_vgg16()
            
        # Remove DataParallel wrapper if exists
        if selected_model_name in ["InceptionV3", "ResNet50"]:
            if isinstance(selected_model, torch.nn.DataParallel):
                selected_model = selected_model.module
                
    st.write(f"### You selected: {selected_model_name}")

# --- Only proceed if a model is selected ---
if selected_model:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.write("### OR:")

    # --- Gallery ---
    gallery = {
        "BAS": ['BA_580.jpg', 'BA_19779.jpg', 'BA_20201.jpg'],
        "EOS": ['EO_29763.jpg', 'EO_24568.jpg', 'EO_25085.jpg'],
        "EBO": ['ERB_168152.jpg', 'ERB_170062.jpg', 'ERB_174098.jpg'],
        "IG": ['PMY_901117.jpg', 'MMY_630078.jpg', 'MY_318125.jpg'],
        "LYT": ['LY_742481.jpg', 'LY_731097.jpg', 'LY_743393.jpg'],
        "MON": ['MO_849518.jpg', 'MO_888999.jpg', 'MO_912563.jpg'],
        "NGS": ['SNE_746083.jpg', 'BNE_378921.jpg', 'SNE_790562.jpg'],
        "PLA": ['PLATELET_969782.jpg', 'PLATELET_37710.jpg', 'PLATELET_815342.jpg']
    }

    # --- Class Selection ---
    class_choice = st.selectbox("Choose a class", ["All"] + list(gallery.keys()))

    # --- Decide which images to show ---
    labels, paths = [], []
    base_dir = './Streamlit/gallery/'
    if class_choice == "All":
        for label, imgs in gallery.items():
            for img_name in imgs:
                labels.append(label)
                paths.append(f"{base_dir}{img_name}")
    else:
        for img_name in gallery[class_choice]:
            labels.append(class_choice)
            paths.append(f"{base_dir}{img_name}")

    # --- Show images in gallery expander ---
    selected_gallery = None
    with st.expander("Show Gallery", expanded=False):
        loaded_images = [Image.open(p) for p in paths]
        selected_gallery = image_select(
            label=f"Choose from {class_choice} gallery",
            images=loaded_images,
            captions=labels,
            use_container_width=False
        )

    # --- Store and retrieve selection in session state ---
    if selected_gallery is not None:
        st.session_state["selected_gallery"] = selected_gallery
    selected_gallery = st.session_state.get("selected_gallery", None)

    # --- Load selected image ---
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB").resize((299, 299))
    elif selected_gallery is not None:
        # selected_gallery is now already a PIL Image object
        image = selected_gallery.convert("RGB").resize((299, 299))

    # --- Image Prediction and Display ---
    if image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Image")

            # --- Preprocess and predict ---
            if "VGG16" in selected_model_name:
                img_array = np.expand_dims(np.array(image), axis=0)
                img_tensor = preprocess_input(img_array)
                predict_acc = selected_model.predict(img_tensor, verbose=0)
                pred = np.argmax(predict_acc, axis=-1)[0]
                pred_prob = predict_acc[0, pred]
            else:
                weights = Inception_V3_Weights.DEFAULT if 'InceptionV3' in selected_model_name else ResNet50_Weights.DEFAULT
                preprocess = weights.transforms()
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = selected_model(img_tensor)
                    pred = output.argmax(dim=1).item()
                    pred_prob = F.softmax(output, dim=1)[0, pred].item()

            st.write(f"## Prediction: {class_names[pred]}")
            st.write(f"## Probability: {pred_prob*100:.2f}%")

        # --- Processed Image Display ---
        with col2:
            if 'VGG16' in selected_model_name:
                img_display = img_tensor[0][..., ::-1]
                img_display = np.clip(img_display, 0, 255) / 255.0
                st.image(img_display, caption="Processed Image")
            else:
                img_np = img_tensor.squeeze().permute(1, 2, 0).numpy().clip(0, 1)
                st.image(img_np, caption="Processed Image")

        # --- Grad-CAM ---
        generate_cam = st.button("Generate Grad-CAM")
        if generate_cam:
            if 'VGG16' in selected_model_name:
                pred_class = np.argmax(predict_acc[0])
                heatmap = make_gradcam_heatmap_keras(img_tensor, selected_model, 'block5_conv3', pred_class)
            else:
                target_layer = "Mixed_7c" if 'InceptionV3' in selected_model_name else "layer4"
                heatmap, _ = make_gradcam_heatmap(img_tensor, selected_model, target_layer_name=target_layer)

            img_np = np.array(image)
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.4 * heatmap_color + 0.6 * get_canny_edge(img_np/255.0)
            overlay = np.clip(overlay, 0, 1)
            st.image(overlay, caption="Grad-CAM Result")
        else:
            st.empty()
