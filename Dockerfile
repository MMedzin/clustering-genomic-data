FROM python:3.9.0
RUN mkdir clustering-genomic-data
WORKDIR /clustering-genomic-data
COPY requirements.txt /clustering-genomic-data/requirements.txt
RUN pip install -r requirements.txt
ENTRYPOINT ["/bin/bash"]