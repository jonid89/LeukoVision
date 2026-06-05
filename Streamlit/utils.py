import torch
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

def make_gradcam_heatmap(img_tensor, model, target_layer_name, pred_index=None):   
    """
    Generates a Grad-CAM heatmap for a PyTorch model using hooks.
    """
    # We'll store the gradients and activations in these variables
    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    target_layer = dict(model.named_modules())[target_layer_name]
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_handle)

    # 1. Forward pass to get model output
    output = model(img_tensor)

    # If no prediction index is provided, use the one with the highest probability
    if pred_index is None:
        pred_index = output.argmax(dim=1).item()
    
    # 2. Backward pass to get gradients
    model.zero_grad()
    output[0, pred_index].backward()

    # Remove hooks after use
    forward_handle.remove()
    backward_handle.remove()

    # 3. Pool gradients and compute heatmap
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the channels of the activation map
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    return heatmap, pred_index

def make_gradcam_heatmap_keras(img_array, model, last_conv_layer_name='block5_conv3', pred_index=None):
    """
    Generates a Grad-CAM heatmap for a Keras/TensorFlow model.
    """
    # build a model that maps inputs -> (last_conv_output, model_output)
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # gradients of the target class w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)  # shape: (1, H, W, C)

    # global average pooling on gradients -> importance for each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # remove batch dim
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    #heatmap = tf.reduce_sum(conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)

    # post-process
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def get_canny_edge(img, threshold1=30, threshold2=80):
    """
    Function to get the canny edge of an image
    Input: img in (H, W, 3), dtype uint8 or float [0,1]
    Output: edge map (H, W, 3), float in [0,1]
    """
    # If float [0,1], convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get the edge (invert edges so white = edge)
    edge = 255 - cv2.Canny(gray, threshold1, threshold2)

    # Convert to 3-channel float [0,1]
    edge = np.stack([edge]*3, axis=-1) / 255.0
    return edge

def white_bg(image):
    img = Image.open(image).convert("RGBA")

    # Create white background
    white_bg = Image.new("RGBA", img.size, "WHITE")
    white_bg.paste(img, (0, 0), img)
    return white_bg