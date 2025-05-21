```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the question-answering pipeline using a DistilBERT model
# DistilBERT is faster and lighter than BERT, suitable for quick responses
question_answerer = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/answer', methods=['POST'])
def get_answer():
    """
    API endpoint to answer questions based on a provided context.
    Expects a JSON payload containing 'question' and 'context' keys.
    Returns a JSON response with the 'answer'.
    """
    data = request.get_json()
    question_text = data.get('question')
    context_text = data.get('context')
    
    # Generate the answer using the question-answering pipeline
    result = question_answerer(question=question_text, context=context_text)
    answer_text = result['answer']
    
    return jsonify({'answer': answer_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
```