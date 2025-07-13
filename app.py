from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from model.model_architecture import LiarModel
import os

app = Flask(__name__)

# Initialize model
def load_model():
    model = LiarModel(
        vocab_size=30522,
        embedding_dim=128,
        n_filters=128,
        filter_sizes=[3,4,5],
        output_dim=6,
        dropout=0.5,
        padding_idx=0,
        input_dim=60,
        input_dim_metadata=6,
        hidden_dim=64,
        n_layers=1,
        bidirectional=True
    )
    model_path = os.path.join('model', 'LIAR2 (2).pt')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

LABEL_MAP = {
    0: "Pants on Fire",
    1: "False", 
    2: "Barely True",
    3: "Half True",
    4: "Mostly True",
    5: "True"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    statement = request.form.get('news_text', '').strip()
    if not statement:
        return render_template('index.html', error="Please enter a statement")
    
    try:
        inputs = tokenizer(
            statement,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        metadata_text = torch.zeros((1, 60), dtype=torch.long)
        metadata_number = torch.randn(1, 6)
        justification = torch.zeros((1, 512), dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(
                inputs['input_ids'],
                metadata_text,
                metadata_number,
                justification
            )
            probs = torch.softmax(outputs, dim=1)[0] * 100
        
        results = []
        for i, prob in enumerate(probs):
            results.append({
                "label": LABEL_MAP[i],
                "confidence": round(prob.item(), 2),
                "type": "Reliable" if i > 2 else "Unreliable"
            })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return render_template('result.html', results=results)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)