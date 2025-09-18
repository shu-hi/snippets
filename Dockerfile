FROM python:3.10-slim

WORKDIR /app

# ① 変更頻度の低い requirements を先に処理
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ② 残りのファイル（models 除外済み）を一括コピー
COPY ./api .

# Flaskポートを公開
EXPOSE 5000

# アプリを実行
CMD ["uvicorn", "main:main", "--host", "0.0.0.0", "--port", "5000"]

