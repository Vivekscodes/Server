from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load a distilled model for question-answering
question_answerer = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.route('/answer', methods=['POST'])
def get_fast_answer():
    data = request.json
    question = data.get('question')
    context = data.get('context')
    
    # Get the answer quickly using the distilled model
    result = question_answerer(question=question, context=context)
    return jsonify({'answer': result['answer']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)