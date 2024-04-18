from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the vectorizer and model
vectorizer = joblib.load('model/vectorizer.pkl')
model = joblib.load('model/spamModel.pkl')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Transform the input text
    X = vectorizer.transform([text])
    X = X.toarray()
    # Make the prediction
    prediction = model.predict(X)
    if prediction[0] == 0:
        result = "The given text is not spam."
    else:
        result = "The given text is spam."
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)