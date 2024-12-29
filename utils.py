import re
import pandas as pd
import numpy as np
from pyvi import ViTokenizer  

import os 
import pickle
import json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns

import sys

# Sử dụng os.getcwd() để lấy thư mục gốc của dự án
sys.path.append(os.path.join(os.getcwd(), 'correct_word_model'))

from correct_word_model.tool.predictor import Predictor

# Lấy ra mô hình sửa lỗi chính tả của buiquangmanhhp1999 trong bài VietnameseOcrCorrection
model_predictor = Predictor(device='cpu', model_type=r'seq2seq', weight_path=r'correct_word_model/weights/seq2seq_0.pth')

# Định nghĩa thư mục lưu trữ
data_path = r'./models'

# Hàm đọc các file dictionary để sửa nội dung văn bản
def doc_dictionary(data_path,file_name):
    # Đọc file JSON
    with open(f"{data_path}/{file_name}.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def normalize_acronyms(text, replace_list):
    for k, v in replace_list.items():
        text = text.replace(k, v)
    return text

# Hàm tiền xử lý văn bản
def remove_non_mean_word(text):
    sentences = []
    non_mean_words = []
    texts = text.split(' ')
    for text in texts:
        if len(text) < 8:  # Ví dụ, nếu chuỗi có độ dài nhỏ hơn 8 thì có thể bỏ qua
            sentences.append(text)
        else:
            non_mean_words.append(text)
    text = ' '.join(sentences)
    
    return text

# Hàm tiền xử lý văn bản
def preprocess_text(text, data_path):
    text = str(text)
    
    # Chuyển thành chữ thường
    text = text.lower()

    # Loại bỏ ký tự kéo dài 3 lần trở lên
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Loại bỏ URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Xóa số
    text = re.sub(r"\d+", "", text)  
    
    # Xóa bỏ ký tự đặc biệt
    special_characters = [':', ',', '.', "'", '"', '!', '?', '-', '_', '(', ')', '[', ']', '{', '}', '&', '@', '#', '$', '%', '^', '*', '+', '=', '|', '\\', '/', '<', '>', '~', '`', '‘', '’', '“', '”', '•', '…', ';', '؛']

    pattern = f"[{re.escape(''.join(special_characters))}]"
    
    # Thay thế tất cả các ký tự đặc biệt bằng khoảng trắng
    text = re.sub(pattern, ' ', text)

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    
    # Thêm khoảng trắng vào đầu và cuối text
    text = ' ' + text + ' '  
    
    # Chuẩn hóa từ sai chính tả
    file_names = ['sua_viet_tat','sua_chinh_ta','dich_tieng_anh']
    for file_name in file_names:
        text = normalize_acronyms(text, doc_dictionary(data_path,file_name))  
    
    # Xóa các ký tự không có ý nghĩa với điều kiện dài hơn 7 kí tự
    text = remove_non_mean_word(text)
    
    # Loại bỏ ký tự kéo dài 2 lần trở lên
    text = re.sub(r"(.)\1{1,}", r"\1", text)    
    
    # Chuẩn hóa các từ sai chính tả bằng model
    try:
        text = model_predictor.predict(text)
    except (KeyError, ValueError):
        pass

    # Thay thế các icon và emoji thành từ miêu tả nó
    emoji_list = ['UNICODE_EMO_VI', 'EMOTICONS_VI']    
    for emoji in emoji_list:
        dict_emoji = doc_dictionary(data_path, emoji)
        for emoji, description in dict_emoji.items():
            text = text.replace(emoji, description)
    
    # Chuyển thành chữ thường        
    text = text.lower()
    
    # Xóa bỏ ký tự đặc biệt    
    special_characters = [':', ',', '.', "'", '"', '!', '?', '-', '_', '(', ')', '[', ']', '{', '}', '&', '@', '#', '$', '%', '^', '*', '+', '=', '|', '\\', '/', '<', '>', '~', '`', '‘', '’', '“', '”', '•', '…', ';', '؛']
    pattern = f"[{re.escape(''.join(special_characters))}]"
    
    # Thay thế tất cả các ký tự đặc biệt bằng khoảng trắng
    text = re.sub(pattern, ' ', text)

    # Tách từ 
    text = ViTokenizer.tokenize(text)
    
    return text

def tao_mo_hinh(data_path):
    # Đọc cấu trúc mô hình từ file JSON
    with open(f'{data_path}/model_structure.json', "r") as json_file:
        model_json = json_file.read()

    # Tạo mô hình từ cấu trúc JSON
    model = model_from_json(model_json)

    # Đọc trọng số mô hình từ file pickle
    with open(f'{data_path}/model_weights.pkl', "rb") as weights_file:
        model.set_weights(pickle.load(weights_file))

    # Tải tokenizer từ file
    with open(f'{data_path}/tokenizer.pkl', "rb") as f:
        tokenizer = pickle.load(f)

    max_length = 60
    return model, tokenizer, max_length

def predict_sentiment(model, tokenizer, input_texts, max_length):
    """
    Hàm dự đoán cảm xúc từ văn bản.
    
    Args:
        model: Mô hình đã huấn luyện.
        tokenizer: Bộ tokenizer đã sử dụng để huấn luyện mô hình.
        text: Văn bản đầu vào cần dự đoán (chuỗi hoặc danh sách chuỗi).
        max_length: Độ dài tối đa đã sử dụng khi pad chuỗi.

    Returns:
        Lớp cảm xúc dự đoán (giá trị từ 1 đến 5).
    """
    if isinstance(input_texts, str):  # Nếu đầu vào là một chuỗi
        text = [input_texts]
    elif isinstance(input_texts, list):  # Nếu là danh sách các chuỗi
        text = input_texts
        
    # Tokenize và pad chuỗi
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen = max_length, padding = 'post', truncating = 'post')

    # Dự đoán
    predictions = model.predict(padded_sequences)
    
    # # Trả về lớp cảm xúc dự đoán (giá trị từ 1 đến 5)
    # predicted_labels = np.argmax(predictions, axis=1) + 1  # Shift lại từ [0, 4] về [1, 5]
    
    # Danh sách nhãn cảm xúc
    labels = ["Rất tiêu cực", "Tiêu cực", "Trung tính", "Tích cực", "Rất tích cực"]
    nhan_so_sao = ['1','2','3','4','5']
    # Ánh xạ predicted_labels thành các nhãn cảm xúc
    predicted_label = np.argmax(predictions, axis=1)[0]
    emotional_label = labels[predicted_label]
    so_sao = nhan_so_sao[predicted_label]  
    # Chuyển đổi các xác suất thành phần trăm
    predictions_percentage = predictions[0] * 100  # Chuyển xác suất thành phần trăm
    percent_result = max(predictions_percentage)  # Lấy phần trăm cao nhất

    return emotional_label, percent_result, so_sao



def predict_sentiment_multi(model, tokenizer, input_text, max_length):
    """
    Hàm dự đoán cảm xúc từ văn bản và trả về thống kê chi tiết.

    Args:
        model: Mô hình đã huấn luyện.
        tokenizer: Bộ tokenizer đã sử dụng để huấn luyện mô hình.
        input_text: Văn bản đầu vào cần dự đoán (chuỗi hoặc nhiều dòng).
        max_length: Độ dài tối đa đã sử dụng khi pad chuỗi.

    Returns:
        Một đối tượng res chứa thông tin thống kê kết quả dự đoán.
    """
    if not isinstance(input_text, str):  # Kiểm tra input_text phải là chuỗi
        raise ValueError("Input phải là một chuỗi.")

    # Tách các dòng nếu input là một văn bản chứa nhiều dòng
    input_lines = input_text.split('\n')  # Tách chuỗi thành các dòng

    emotional_labels = []
    percent_results = []

    # Danh sách nhãn cảm xúc
    labels = ["Rất tiêu cực", "Tiêu cực", "Trung tính", "Tích cực", "Rất tích cực"]
    label_counts = {label: 0 for label in labels}  # Khởi tạo số lượng từng nhãn

    # Dự đoán cảm xúc cho từng dòng
    for line in input_lines:
        if line.strip():  # Bỏ qua các dòng trống
            # Tokenize và pad chuỗi
            sequence = tokenizer.texts_to_sequences([line])  # Chuyển thành danh sách 1 phần tử
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

            # Dự đoán
            predictions = model.predict(padded_sequence)

            # Xác định nhãn cảm xúc dự đoán
            predicted_label_idx = np.argmax(predictions, axis=1)[0]  # Lấy giá trị duy nhất
            emotional_label = labels[predicted_label_idx]

            # Xác suất dự đoán cao nhất, làm tròn đến số thập phân thứ nhất
            percent_result = round(np.max(predictions) * 100, 2)

            # Thêm kết quả vào danh sách
            emotional_labels.append(emotional_label)
            percent_results.append(percent_result)

            # Tăng số lượng nhãn cảm xúc tương ứng
            label_counts[emotional_label] += 1

    # Tính toán các thống kê
    total_comments = len(percent_results)
    if total_comments == 0:
        return {
            "total_comments": 0,
            "highest_percent": None,
            "lowest_percent": None,
            "average_percent": None,
            "negative_comments": 0,
            "negative_highest_percent": None,
            "negative_lowest_percent": None,
            "negative_average_percent": None,
            "positive_comments": 0,
            "positive_highest_percent": None,
            "positive_lowest_percent": None,
            "positive_average_percent": None,
            "label_counts": json.dumps({})  # Trả về JSON rỗng
        }

    highest_percent = max(percent_results)
    lowest_percent = min(percent_results)
    average_percent = round(sum(percent_results) / total_comments, 2)

    # Phân loại cảm xúc
    negative_indices = [i for i, label in enumerate(emotional_labels) if label in ["Rất tiêu cực", "Tiêu cực"]]
    positive_indices = [i for i, label in enumerate(emotional_labels) if label in ["Tích cực", "Rất tích cực"]]

    negative_comments = len(negative_indices)
    positive_comments = len(positive_indices)

    negative_highest_percent = max([percent_results[i] for i in negative_indices], default=None)
    negative_lowest_percent = min([percent_results[i] for i in negative_indices], default=None)
    negative_average_percent = (
    round(
        sum([percent_results[i] for i in negative_indices]) / negative_comments,
        2
    ) if negative_comments > 0 else None
)

    positive_highest_percent = max([percent_results[i] for i in positive_indices], default=None)
    positive_lowest_percent = min([percent_results[i] for i in positive_indices], default=None)
    positive_average_percent = (
    round(
        sum([percent_results[i] for i in positive_indices]) / positive_comments,
        2
    ) if positive_comments > 0 else None
)

    # Chuyển label_counts sang JSON
    label_counts_json = json.dumps(label_counts, ensure_ascii=False)

    # Trả về kết quả
    res = {
        "total_comments": total_comments,
        "highest_percent": highest_percent,
        "lowest_percent": lowest_percent,
        "average_percent": average_percent,
        "negative_comments": negative_comments,
        "negative_highest_percent": negative_highest_percent,
        "negative_lowest_percent": negative_lowest_percent,
        "negative_average_percent": negative_average_percent,
        "positive_comments": positive_comments,
        "positive_highest_percent": positive_highest_percent,
        "positive_lowest_percent": positive_lowest_percent,
        "positive_average_percent": positive_average_percent,
        "label_counts": label_counts_json  # Số lượng từng nhãn cảm xúc dưới dạng JSON
    }

    return res

model, tokenizer, max_length = tao_mo_hinh(data_path)


def process_input_texts(input_texts, model, tokenizer, max_length, data_path):
    # Khởi tạo danh sách để lưu kết quả
    list_text = []

    # Lặp qua từng câu trong input_texts (chia cách bởi dòng mới)
    for text in input_texts.splitlines():  # Split text vào các dòng riêng biệt
        # Tiền xử lý và chuẩn hóa văn bản
        text = preprocess_text(text, data_path)
        
        # Dự đoán nhãn cảm xúc và tỉ lệ phần trăm
        label_result, percent_result, so_sao = predict_sentiment(model, tokenizer, text, max_length)
        
        # In ra kết quả cho câu
        # print(f'Có {percent_result} là {label_result} cho câu: {text}')
        
        # Lưu kết quả vào list_text
        list_text.append({
            'Bình luận': text,
            'Số sao': so_sao,
            'Nhãn': label_result        
        })
    
    # Trả về danh sách kết quả
    return list_text


# Kỹ thuật TF-IDF
def extract_top_tfidf_words_per_rating(dataset, num_words=10):
    # Tạo TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2),max_features=5000, stop_words=None)  # Giới hạn số từ đặc trưng
    
    # Lưu từ khóa phổ biến nhất theo từng loại sao
    top_words_by_rating = {}
    
    for rating in sorted(dataset['Số sao'].unique()):
        # Lọc dữ liệu theo rating
        filtered_data = dataset[dataset['Số sao'] == rating]
        
        # Tính TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['Bình luận'])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        # Lấy giá trị trung bình TF-IDF của mỗi từ
        tfidf_mean = np.mean(tfidf_matrix.toarray(), axis=0)
        tfidf_scores = pd.DataFrame({'word': feature_names, 'score': tfidf_mean})
        
        # Lấy top từ phổ biến nhất
        top_words = tfidf_scores.sort_values(by='score', ascending=False).head(num_words)
        top_words_by_rating[rating] = top_words
    
    return top_words_by_rating


def plot_pie_and_tfidf_bars(top_words_by_rating, dataset, data_path):
    # Đếm số lượng đánh giá theo từng loại sao
    sentiment_labels = {
    "1": "Rất tiêu cực",
    "2": "Tiêu cực",
    "3": "Trung lập",
    "4": "Tích cực",
    "5": "Rất tích cực"
    }

    sentiment_colors = {
        "Rất tiêu cực": "#e60000",  # Màu đỏ
        "Tiêu cực": "#e7811a",      # Màu cam
        "Trung lập": "#e8c268",     # Màu vàng
        "Tích cực": "#eb885f",      # Màu xanh lá nhạt
        "Rất tích cực": "#6dd94a"   # Màu xanh lá đậm
    }
    # Đếm tần suất của các đánh giá (số sao)
    rating_counts = dataset['Số sao'].value_counts().sort_index()
    
    # Lọc các mức đánh giá thực tế có trong dữ liệu
    filtered_sentiment_labels = {k: sentiment_labels[k] for k in rating_counts.index}
    
    # Lấy số lượng đánh giá cho từng cảm xúc
    sentiment_counts = [rating_counts.get(rating, 0) for rating in sorted(filtered_sentiment_labels.keys())]

    # Tạo nhãn với tổng số bình luận
    labels_with_counts = [
        f"{filtered_sentiment_labels[rating]}"#\n ({rating_counts.get(rating, 0)} bình luận)"
        for rating in sorted(filtered_sentiment_labels.keys())
    ]
    # Giả sử dataset là DataFrame của bạn
    num_charts = len(dataset['Số sao'].unique()) + 1
    # Tạo biểu đồ tròn ở ô đầu tiên
    # Xác định số cột và số hàng
    if num_charts == 2:
        ncols, nrows = 2, 1  # 2 biểu đồ: 2 cột, 1 hàng
    elif num_charts == 3:
        ncols, nrows = 3, 1  # 3 biểu đồ: 3 cột, 1 hàng
    elif num_charts == 4:
        ncols, nrows = 2, 2  # 4 biểu đồ: 2 cột, 2 hàng
    elif num_charts in [5, 6]:
        ncols, nrows = 3, 2  # 5 hoặc 6 biểu đồ: 3 cột, 2 hàng
    fig = plt.figure(figsize = (20, 12))
    
    if num_charts in [4, 5, 6]:
        gs = GridSpec(nrows = nrows + 1, ncols = ncols, figure = fig, height_ratios = [0.28, 2, 2])  
    else:
        gs = GridSpec(nrows = nrows + 1, ncols = ncols, figure = fig, height_ratios = [0.28, 4])    

    axes = []
    if num_charts == 5:
        axes = [fig.add_subplot(gs[3]), 
                fig.add_subplot(gs[1,1:3]),
        fig.add_subplot(gs[6]), 
        fig.add_subplot(gs[7]),
        fig.add_subplot(gs[8])]
    else:
        for i in range(ncols * nrows):
            axes.append(fig.add_subplot(gs[i + ncols]))


    axs = fig.add_subplot(gs[0:ncols])
    axs.axis('off')
    # if title != None:
    #     axs.set_title(title, ha='center', va='center')
    axs.text(0.5, 0.5, str(f'DASHBOARD PHÂN TÍCH CẢM XÚC BÌNH LUẬN KHÁCH HÀNG'), fontsize=20, ha='center', va='center')
    axs.add_patch(plt.Rectangle((0, 0.2), 1, 0.7, fill=None, edgecolor='black', linewidth=2))

    # Lấy màu từ sentiment_colors theo từng nhãn
    colors_for_pie = [
        sentiment_colors[filtered_sentiment_labels[rating]]
        for rating in sorted(filtered_sentiment_labels.keys())
    ]
    
    # Hàm format cho biểu đồ tròn
    def autopct_func(pct, allvals):
        absolute = int(pct/100.*sum(allvals))
        return f"{pct:.1f}% ({absolute})"
        #return f"{pct:.1f}% ({absolute} bình luận)"

    
    # Sử dụng cmap để áp dụng màu gradient
    axes[0].pie(
        sentiment_counts,
        labels=labels_with_counts,
        colors=colors_for_pie,  # Sử dụng màu trực tiếp từ sentiment_colors
        autopct=lambda pct: autopct_func(pct, sentiment_counts),
        startangle=90,
    )
    axes[0].set_title("Tỉ lệ số lượng đánh giá theo cảm xúc", fontsize=16)
    # axes[0].axis('off')

    # Màu sắc cảm xúc
    sentiment_colors = {
        "Rất tiêu cực": "#e60000",  # Màu đỏ
        "Tiêu cực": "#e7811a",      # Màu cam
        "Trung lập": "#e8c268",     # Màu vàng
        "Tích cực": "#eb885f",      # Màu xanh lá nhạt
        "Rất tích cực": "#6dd94a"   # Màu xanh lá đậm
    }

    custom_order = ['1', '5', '3', '2' ,'4']
    ax_index = 1
    for index, rating in enumerate(custom_order):
        if rating in dataset['Số sao'].values:
            # if rating in top_words_by_rating:
                words = top_words_by_rating[rating]
                
                # Lấy màu từ sentiment_colors
                sentiment_label = sentiment_labels[rating]
                sentiment_color = sentiment_colors.get(sentiment_label, "#FFFFFF")  # Màu mặc định là trắng nếu không có màu trong từ điển

                # Tạo gradient màu từ màu sắc đã cho
                gradient_palette = sns.light_palette(sentiment_color, as_cmap=True)  # Tạo gradient từ màu sắc

                # Áp dụng màu gradient cho các thanh (bar)
                sns.barplot(
                    data=words,
                    x='score',
                    y='word',
                    palette=gradient_palette(np.linspace(1, 0.4, len(words))),  # Dùng gradient từ 0.3 đến 1 để tránh quá tối
                    ax=axes[ax_index],
                    orient='h',
                )
                axes[ax_index].set_title(
                    f"Đánh giá {sentiment_labels[rating]}",
                    fontsize=16
                    
                )
                axes[ax_index].set_xlabel("TF-IDF Score", fontsize=12)
                axes[ax_index].set_ylabel("")  # Loại bỏ nhãn cột y
                ax_index = ax_index + 1


    plt.tight_layout()
    plt.savefig(f"./static/dashboard.png")  # Chỉ định đường dẫn thư mục và tên file


