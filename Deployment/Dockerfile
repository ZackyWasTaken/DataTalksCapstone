FROM python:3.10-slim

ENV PYTHONWARNINGS="all:::0"

WORKDIR /app

RUN pip install --upgrade pip

COPY ["Pipfile", "./"]

RUN pip install pipenv

RUN pipenv install --deploy

COPY ["xception_final_139_0.820_0.936.h5", "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]