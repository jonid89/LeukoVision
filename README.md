# LeukoVision: Identifying White Blood Cells from Microscopic Images 🩸

This project aims to build deep learning models to automatically classify different types of white blood cells (WBCs). We leveraged state-of-the-art architectures including **InceptionV3**, **ResNet50**, and **VGG16** to accurately identify various blood cell types.

## Live Demo 🚀
Check out the live application running on Streamlit Community Cloud: 
[**Launch Streamlit App**](https://leukovision-ai.streamlit.app/)

Use the **Vision Lab** inside the app to upload your own blood smear images, test our models, and visualize what the AI "sees" using Grad-CAM heatmaps.

## How to Use Locally 💻

If you want to run the project or use the models on your own machine:

1. **Install Dependencies:** We recommend using the provided `requirements.txt` file to install all necessary Python packages in a virtual environment to avoid conflicts:
   ```bash
   pip install -r requirements.txt
   ```
2. **Load Models:** You can load our trained models (`.h5` for VGG16, `.pth` for PyTorch models) directly into your scripts for inference.
3. **Run Streamlit:** Navigate to the root directory and start the app locally:
   ```bash
   streamlit run Streamlit/LeukoVision.py
   ```

## Folder Structure 📂

```text
root/ 
├── Notebooks/          # Jupyter notebooks for data preprocessing and model training
│   ├── inceptionv3/
│   ├── resnet50/
│   └── vgg16/
├── Report/             # Documentation, papers, and final reports
├── Streamlit/          # Streamlit app files, pages, and gallery assets
├── src/                # Python scripts for support functions and image checking
├── requirements.txt    # Required dependencies
└── README.md           # Project overview
```

## Data Reference 📚
The data used as a reference for training our models was introduced in the following publication:
- Acevedo et al. (2019) - *Computer Methods and Programs in Biomedicine*

The dataset is publicly available to download here:
- Mendeley Data: Blood Cell Dataset