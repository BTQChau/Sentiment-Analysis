from flask import Flask, render_template, request
from utils import predict_sentiment, model, tokenizer, predict_sentiment_multi, extract_top_tfidf_words_per_rating, plot_pie_and_tfidf_bars, data_path, process_input_texts
import json
import pandas as pd

app = Flask(__name__)

@app.route("/") 
def home():
    return render_template("index.html")

@app.route("/dashboard") 
def dashboard():
    return render_template("dashboard.html")

@app.route("/single") 
def single():
    return render_template("index.html")

@app.route("/multiple") 
def multiple():
    return render_template("multi_analysis.html")

@app.route("/view-details")
def view_details():
     # Lấy dữ liệu từ query string
    max_length = 60
    message = request.args.get("message", "")
    list_texts = process_input_texts(message, model, tokenizer, max_length, data_path)
    
    df_tfidf = pd.DataFrame(list_texts)
    top_words_by_rating = extract_top_tfidf_words_per_rating(df_tfidf, num_words=15)
    plot_pie_and_tfidf_bars(top_words_by_rating, df_tfidf, data_path)

    print("message", message)
    return render_template("dashboard.html")

@app.route('/analysis', methods=['POST'])
def analysis():
    max_length = 60
    comment = request.form['comment']

    emotional_label, percent_result, so_sao = predict_sentiment(model, tokenizer, comment, max_length)
    print(emotional_label, percent_result)
    return render_template ("index.html",
                            comment=comment,
                            emotional_label = emotional_label,
                            percent_result = percent_result)


@app.route('/analysis-multi', methods=['POST'])
def analysis_multi():
    max_length = 60
    message = request.form['message']  # Lấy dữ liệu từ textarea

    # Gọi hàm dự đoán
    res = predict_sentiment_multi(model, tokenizer, message, max_length)
    print("res", res)  # Debug để kiểm tra nội dung của res

    # Truyền giá trị từ dictionary vào template
    return render_template(
        "multi_analysis.html",
        total_comments=res["total_comments"],
        highest_percent=res["highest_percent"],
        lowest_percent=res["lowest_percent"],
        average_percent=res["average_percent"],
        negative_comments=res["negative_comments"],
        negative_highest_percent=res["negative_highest_percent"],
        negative_lowest_percent=res["negative_lowest_percent"],
        negative_average_percent=res["negative_average_percent"],
        positive_comments=res["positive_comments"],
        positive_highest_percent=res["positive_highest_percent"],
        positive_lowest_percent=res["positive_lowest_percent"],
        positive_average_percent=res["positive_average_percent"],
        label_counts=res["label_counts"],
        isHidden = 1,
        message=message
    )

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port=8080, debug=True)


