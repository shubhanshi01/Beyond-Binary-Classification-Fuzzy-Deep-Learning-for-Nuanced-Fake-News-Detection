import torch
try:
    model = torch.load('model/model_weights.pkl', map_location='cpu')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")