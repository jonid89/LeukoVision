import streamlit as st
from streamlit_option_menu import option_menu
import sys
from utils import white_bg
import io
from tensorflow.keras.applications import InceptionV3
import numpy as np
import pandas as pd


st.title('Modeling')

st.markdown('LeukoVision leverages state-of-the-art convolutional neural networks (CNNs) ' \
'to classify different types of blood cells. These models—InceptionV3, ResNet50, and VGG16—have ' \
'been widely used in medical image analysis due to their ability to capture subtle patterns in ' \
'microscopy images.')

section = option_menu(
    menu_title=None,
    options=["InceptionV3", "ResNet50", "VGG16"],  
    icons=['🔬','🧬','🧪'],
    orientation="horizontal",
)

# section = st.sidebar.radio(
#     "Choose Section",
#     ["InceptionV3", "ResNet50", "VGG16"]
# )

if section == "InceptionV3":
    st.subheader("InceptionV3 🔬")
    st.markdown(""" ### Overview
    <div style="text-align: justify;">
                
    InceptionV3 is a deep convolutional neural network architecture designed for efficient and accurate image recognition. It is an evolution of the original GoogLeNet (Inception) model, optimized for both computational efficiency and high performance on large-scale image classification tasks.  

    InceptionV3 achieves high accuracy on benchmark datasets such as ImageNet, while keeping computational resources manageable. This makes it a popular choice for real-world applications, including medical imaging, object detection, and visual recognition tasks.
                
    </div>
    """,unsafe_allow_html=True)

    st.image(white_bg('./Streamlit/pages/images/inception/inceptionv3.png'), caption='Architecture diagram of InceptionV3',use_container_width=True)
    model = InceptionV3(weights='imagenet')
    with st.expander("See Full Model Summary"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.code(stream.getvalue())
    st.markdown('### Performace')
    st.markdown(""" <div style="text-align: justify;">
    The training accuracy approaches 99.97%, while the validation accuracy reaches 
    98.29%, demonstrating the model’s strong ability to accurately classify different cell types.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show InceptionV3 loss and accuracy plot"):
        st.image(white_bg('./Streamlit/pages/images/inception/loss_acc.png'), caption='Loss and accruacy plot from InceptionV3 training',use_container_width=True)
    st.markdown(""" <div style="text-align: justify;">
                The test set shows a very high accuracy of 98.28% 
                and that is reflected in the digonal form of the confusion matrix.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show InceptionV3 confusion matrix"):
        st.image(white_bg('./Streamlit/pages/images/inception/cm.png'), caption='Confusion matrix of InceptionV3 test set',use_container_width=True)
    report = pd.read_csv("./Streamlit/pages/images/inception/class_report.txt", 
                     sep="\s+", header=0,
                     names=["Class", "Recall", "Specificity", "Precision", "F1-Score"])
    report.index = report.index + 1
    numeric_cols = report.select_dtypes(include="number").columns
    styled = report.style.format({col: "{:.2f}" for col in numeric_cols}) \
                        .set_properties(**{"text-align": "center"}) \
                        .set_table_styles([{
                            "selector": "th",
                            "props": [("text-align", "center"), ("font-weight", "bold")]
                        }])
    st.markdown(""" <div style="text-align: justify;">
                The classification metrics consistently range between 0.96 and 1.00, 
                demonstrating that the InceptionV3 model performs exceptionally well in 
                distinguishing among different WBC subtypes. This highlights both the robustness of 
                the model and its suitability for automated cell classification tasks.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show classification report"):
        st.dataframe(styled)
    st.markdown('### Interpretability by Grad-CAM')
    st.markdown(""" <div style="text-align: justify;">
Grad-CAM visualization shows that the model consistently focuses on the cell itself while ignoring the 
                background, suggesting that the classification is based on cytomorphological features 
                of the cells rather than irrelevant image artifacts.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show InceptionV3 Grad-CAM results"):
        st.image(white_bg('./Streamlit/pages/images/inception/prediction.png'), 
                 caption='Four randomly selected test set images with their ground-truth '
                 'and predicted labels. (Bottom) Corresponding Grad-CAM visualizations highlighting ' \
                 'the image regions most influential in the model’s predictions.',use_container_width=True)
    

# The key idea behind InceptionV3 is the use of **Inception modules**, which allow the network to capture features at multiple scales simultaneously. Each module applies several convolutions of different sizes in parallel and concatenates the results, enabling the model to learn both fine and coarse features from an image.  

#     InceptionV3 incorporates several advanced techniques to improve training and reduce overfitting, including:

#     - **Factorized convolutions** to reduce computational cost while maintaining performance  
#     - **Auxiliary classifiers** that provide additional gradient signals during training  
#     - **Batch normalization** to stabilize and accelerate training  
#     - **Label smoothing** to improve generalization 

elif section == "ResNet50":
    st.subheader("ResNet50 🧬")
    

elif section == "VGG16":
    st.subheader("VGG16 🧪")
