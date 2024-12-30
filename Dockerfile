# Sử dụng Python 3.9 làm base image
FROM python:3.9

# Bật chế độ không buffer để log xuất ngay lập tức
ENV PYTHONUNBUFFERED True

# Thiết lập thư mục làm việc
ENV APP_HOME /app
WORKDIR $APP_HOME

# Cài đặt git và git-lfs
RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install

# Copy code vào container
COPY . ./

# Nếu sử dụng Git LFS để quản lý tệp lớn, tải tệp LFS trong quá trình build
RUN git lfs pull

# Cài đặt các dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app