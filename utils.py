import pickle
import os 
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

# Định nghĩa thư mục lưu trữ
output_dir = r"D:\năm 4 kì 1\KPW\Emotion\models"

# Hàm để lấy đường dẫn file theo tên
def get_file_by_name(directory, filename):
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file '{filename}' trong thư mục '{directory}'")
    return file_path

# Đọc cấu trúc mô hình từ file JSON
json_path = get_file_by_name(output_dir, "model_structure.json")
with open(json_path, "r") as json_file:
    model_json = json_file.read()

# Tạo mô hình từ cấu trúc JSON
model = model_from_json(model_json)

print("Cấu trúc model đã được tải thành công!")

# Đọc trọng số mô hình từ file pickle
weights_path = get_file_by_name(output_dir, "model_weights.pkl")
with open(weights_path, "rb") as weights_file:
    model.set_weights(pickle.load(weights_file))
    
print("Trọng số model đã được tải thành công!")

# Tải tokenizer từ file
tokenizer_path = get_file_by_name(output_dir, "tokenizer.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer đã được tải thành công!")


def predict_sentiment_array(model, tokenizer, input_text, max_length):
    """
    Hàm dự đoán cảm xúc từ văn bản.
    
    Args:
        model: Mô hình đã huấn luyện.
        tokenizer: Bộ tokenizer đã sử dụng để huấn luyện mô hình.
        input_text: Văn bản đầu vào cần dự đoán (chuỗi hoặc nhiều dòng).
        max_length: Độ dài tối đa đã sử dụng khi pad chuỗi.

    Returns:
        Danh sách các lớp cảm xúc và xác suất dự đoán tương ứng.
    """
    if not isinstance(input_text, str):  # Kiểm tra input_text phải là chuỗi
        raise ValueError("Input phải là một chuỗi.")
    
    # Tách các dòng nếu input là một văn bản chứa nhiều dòng
    input_lines = input_text.split('\n')  # Tách chuỗi thành các dòng
    
    emotional_labels = []
    percent_results = []
    
    # Dự đoán cảm xúc cho từng dòng
    for line in input_lines:
        # Tokenize và pad chuỗi
        sequence = tokenizer.texts_to_sequences([line])  # Chuyển thành danh sách 1 phần tử
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

        # Dự đoán
        predictions = model.predict(padded_sequence)

        # Danh sách nhãn cảm xúc
        labels = ["Rất tiêu cực", "Tiêu cực", "Trung tính", "Tích cực", "Rất tích cực"]

        # Xác định nhãn cảm xúc dự đoán
        predicted_label_idx = np.argmax(predictions, axis=1)[0]  # Lấy giá trị duy nhất
        emotional_label = labels[predicted_label_idx]

        # Xác suất dự đoán cao nhất, làm tròn đến số thập phân thứ nhất
        percent_result = round(np.max(predictions) * 100, 2)

        # Thêm kết quả vào danh sách
        emotional_labels.append(emotional_label)
        percent_results.append(percent_result)

    return emotional_labels, percent_results

def predict_sentiment(model, tokenizer, input_text, max_length):
    """
    Hàm dự đoán cảm xúc từ văn bản.
    
    Args:
        model: Mô hình đã huấn luyện.
        tokenizer: Bộ tokenizer đã sử dụng để huấn luyện mô hình.
        input_text: Văn bản đầu vào cần dự đoán (chuỗi).
        max_length: Độ dài tối đa đã sử dụng khi pad chuỗi.

    Returns:
        Lớp cảm xúc dự đoán (giá trị từ 1 đến 5).
    """
    if not isinstance(input_text, str):  # Kiểm tra input_text phải là chuỗi
        raise ValueError("Input phải là một chuỗi.")
        
    # Tokenize và pad chuỗi
    sequence = tokenizer.texts_to_sequences([input_text])  # Chuyển thành danh sách 1 phần tử
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Dự đoán
    predictions = model.predict(padded_sequence)

    # Danh sách nhãn cảm xúc
    labels = ["Rất tiêu cực", "Tiêu cực", "Trung tính", "Tích cực", "Rất tích cực"]

    # Xác định nhãn cảm xúc dự đoán
    predicted_label_idx = np.argmax(predictions, axis=1)[0]  # Lấy giá trị duy nhất
    emotional_label = labels[predicted_label_idx]

    # Xác suất dự đoán cao nhất, làm tròn đến số thập phân thứ nhất
    percent_result = round(np.max(predictions) * 100, 2)

    return emotional_label, percent_result
