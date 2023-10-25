FROM python:3.9.18-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 4000

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]