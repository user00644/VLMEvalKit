# Stage 1: Build stage with all dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies that might be needed by some python packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# 针对 Debian/Ubuntu 基础镜像
# 放在 Dockerfile 最前面，只执行一次
RUN (find /etc/apt -name '*.list' -o -name '*.sources') | \
    xargs -r sed -Ei \
        -e 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' \
        -e 's|http://security.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libglx-mesa0 libglib2.0-0 libsm6 libxext6 \
        libxrender1 libgomp1 mesa-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



RUN ls -l /usr/lib/x86_64-linux-gnu/libGL.so.1 || echo "libGL.so.1 不存在"


# Copy all requirements files first to leverage Docker cache
COPY requirements.txt requirements.txt
COPY requirements/docs.txt requirements/docs.txt
COPY flask_service/requirements.txt flask_service/requirements.txt

# Install all dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements/docs.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r flask_service/requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN apt-get update && apt-get install -y  libglu1-mesa libglx-mesa0 mesa-utils 

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Set the command to run the application
# Using gunicorn for a more robust production server
# We install it here directly
RUN pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "flask_service.app:app"]
