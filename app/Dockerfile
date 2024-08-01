FROM python:3.10

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PORT=${PORT:-8080} \
    PIP_CACHE_DIR=/.cache

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt
RUN pip install label-studio

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

COPY * /app/

EXPOSE 9090
EXPOSE 8080

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]