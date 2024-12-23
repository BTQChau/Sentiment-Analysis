from flask import Flask, render_template, request
from utils import predict_sentiment, model, tokenizer

app = Flask(__name__)

@app.route("/") 
def home():
    return render_template("index.html")

@app.route('/analysis', methods=['POST'])
def analysis():
    max_length = 100
    comment = request.form['comment']

    emotional_label, percent_result = predict_sentiment(model, tokenizer, comment, max_length)
    return render_template ("index.html",
                            comment=comment,
                            emotional_label = emotional_label,
                            percent_result = percent_result)

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port=8080, debug=True)
    

