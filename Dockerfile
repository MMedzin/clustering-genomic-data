FROM python:3.9.0
RUN mkdir datasets experiment
ADD datasets/ datasets/
ADD experiment experiment/
WORKDIR /experiment
RUN pip install -r requirements.txt
CMD ["experiment/run_experiment.py"]
ENTRYPOINT [ "python" ]