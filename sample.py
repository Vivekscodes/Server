```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the question-answering pipeline with a distilled model for speed
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/answer', methods=['POST'])
def get_answer():
    """
    Endpoint to answer questions based on a given context.
    Accepts a JSON payload with 'question' and 'context' keys.
    """
    data = request.get_json()
    question = data.get('question')
    context = data.get('context')
    
    # Obtain the answer using the question-answering pipeline
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
```