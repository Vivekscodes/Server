```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the question-answering pipeline using a DistilBERT model.
# DistilBERT is faster and lighter than BERT, making it suitable for quick responses.
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

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
    try:
        # Get the JSON data from the request
        request_data = request.get_json()

        # Extract the question and context from the JSON data
        question = request_data.get('question')
        context = request_data.get('context')

        # Generate the answer using the question-answering pipeline
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']

        # Return the answer as a JSON response
        return jsonify({'answer': answer})

    except Exception as e:
        # Handle potential errors during processing
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
```