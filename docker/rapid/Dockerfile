FROM nvcr.io/nvidia/pytorch:22.08-py3
WORKDIR /opt/app
RUN git clone https://github.com/ozantezcan/RAPiD-T.git . \
    && git checkout 9ef4a078df654932c1cfc0361c4b376ce6c906b0

COPY src/rapid rapid
COPY src/dataset dataset
ENTRYPOINT ["python"]
CMD []