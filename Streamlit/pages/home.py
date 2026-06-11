import streamlit as st

# st.set_page_config( layout="wide")

st.title("🏠 Welcome to LeukoVision 🩸")

with st.container():
    st.markdown("""
    <div style="text-align: justify;">
                
    **LeukoVision** is an advanced AI-powered platform designed to analyze and classify blood cell images with precision. Use the [🧪Vision Lab](visionlab) page to get a prediction!  
    Leveraging state-of-the-art deep learning models — **InceptionV3**, **ResNet50**, and **VGG16** — the app provides accurate identification of various blood cell types, including basophils, eosinophils, erythroblasts, lymphocytes, monocytes, neutrophils, and platelets.

    With **LeukoVision**, you can: 
    - Upload blood smear images or select from a curated gallery.
    - Predict the correct cell type with confidence scores for each prediction.
    - Visualize **Grad-CAM heatmaps** to understand which regions of the image influenced the model’s decision.
    - Compare the performance of different models for research and educational purposes.

    This interactive tool is perfect for **researchers, educators, and medical professionals** looking to combine computational pathology with intuitive visualization.
    </div>
    """, unsafe_allow_html=True)

st.write("") # Add a bit of spacing

# Display the 3 images in 3 aligned columns
col1, col2, col3 = st.columns(3)

with col1:
    st.image('./Streamlit/gallery/Image processing/Image processing-1.png', use_container_width=True)
with col2:
    st.image('./Streamlit/gallery/Image processing/Image processing-2.png', use_container_width=True)
with col3:
    st.image('./Streamlit/gallery/Image processing/Image processing-3.png', use_container_width=True)
