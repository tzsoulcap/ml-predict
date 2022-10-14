from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'caps'
# Load the model
model = load_model("./model")

class MyForm(FlaskForm):
    text = StringField("Input Text")
    submit = SubmitField("Enter")

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST', 'GET'])
def index():
    text = ""
    op = ""
    form = MyForm()
    if form.validate_on_submit():
        text = form.text.data
        form.text.data = ""
        if text:
            op = predict(text)
    return render_template("predict.html", form = form, text = text, output = op)

# @app.route('/api',methods=['POST'])
def predict(input):
    # Get the data from the POST request.
    # data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    print(input)
    # prediction = model.predict([[np.array(data['exp'])]])
    pred = model.predict([input]).argmax(axis=1)
    print(pred)
    print(pred[0])

    # output = {'output': pred[0]}
    # Take the first value of prediction
    # output = prediction[0]
    return "Toxic" if pred[0] == 1 else "Not Toxic"
    # return jsonify(op=f"{pred[0]}")
if __name__ == '__main__':
    app.run(port=5000, debug=True)
