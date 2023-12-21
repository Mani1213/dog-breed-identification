FROM python:3.10.13
WORKDIR .
COPY . .
RUN pip install -r requirements.txt
EXPOSE  7860
CMD ["python", "Dog-breed-prediction.py"]
