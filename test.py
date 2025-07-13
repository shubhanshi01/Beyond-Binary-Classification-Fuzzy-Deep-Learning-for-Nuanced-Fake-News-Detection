<<<<<<< HEAD
import torch
try:
    model = torch.load('model/model_weights.pkl', map_location='cpu')
    print("Model loaded successfully!")
except Exception as e:
=======
import torch
try:
    model = torch.load('model/model_weights.pkl', map_location='cpu')
    print("Model loaded successfully!")
except Exception as e:
>>>>>>> 0e809e5f55c81c2c0eb5425ef89b1eaad300a8f0
    print(f"Error loading model: {str(e)}")