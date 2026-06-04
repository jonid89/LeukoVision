import os
from tensorflow.keras.models import load_model

# Get absolute paths relative to this script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', '..', 'Streamlit', 'vgg16_model.h5')
shrunk_model_path = os.path.join(current_dir, '..', '..', 'Streamlit', 'vgg16_model_shrunk.h5')

# Load your current bulky model
print("Loading model...")
model = load_model(model_path)

# Save it again without the optimizer state
print("Saving shrunk model...")
model.save(shrunk_model_path, include_optimizer=False)

print("Done! Check the file size of vgg16_model_shrunk.h5")