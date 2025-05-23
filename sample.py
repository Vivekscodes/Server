```python
from flask import Flask, request, jsonify
from transformers import pipeline
from typing import Dict, Any

app = Flask(__name__)

# Load the question-answering pipeline using a DistilBERT model.
# DistilBERT is faster and lighter than BERT, making it suitable for quick responses.
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")


@app.route('/')
def home() -> str:
    """
    Home endpoint.
    Returns a simple greeting.
    """
    return "Hello, World!"


@app.route('/answer', methods=['POST'])
def get_answer() -> jsonify:
    """
    API endpoint to answer questions based on a provided context.
    Expects a JSON payload containing 'question' and 'context' keys.
    Returns a JSON response with the 'answer'.
    """
    try:
        # Get the JSON data from the request
        request_data: Dict[str, str] = request.get_json()

        # Check if request_data is None
        if request_data is None:
            return jsonify({'error': 'Request body must be JSON'}), 400

        # Extract the question and context from the JSON data
        question: str = request_data.get('question')
        context: str = request_data.get('context')

        # Check if question or context is missing
        if not question:
            return jsonify({'error': 'Missing question'}), 400
        if not context:
            return jsonify({'error': 'Missing context'}), 400

        # Generate the answer using the question-answering pipeline
        result: Dict[str, Any] = qa_pipeline(question=question, context=context)
        answer: str = result['answer']

        # Return the answer as a JSON response
        return jsonify({'answer': answer})

    except Exception as e:
        # Handle potential errors during processing
        print(f"Error: {e}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
```