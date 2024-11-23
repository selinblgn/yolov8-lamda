# AWS Lambda Python 3.9 tabanlı imaj kullanımı
FROM public.ecr.aws/lambda/python:3.9

# Gerekli sistem araçlarını güncelle ve kur (opsiyonel)
RUN yum update -y && yum install -y gcc

# Python bağımlılıklarını yüklemek için requirements.txt dosyasını kopyala
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN python3 -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Çalışma dizinini ayarla
WORKDIR ${LAMBDA_TASK_ROOT}

# Uygulama kodlarını ve ONNX modelini kopyala
COPY lambda_function.py .
COPY model/best.onnx /opt/model/best.onnx

# Ortam değişkenlerini ayarla
ENV S3_BUCKET_NAME=yolov8modell \
    ONNX_MODEL_PATH=/opt/model/best.onnx

# Lambda handler komutunu ayarla
CMD ["lambda_function.handler"]
