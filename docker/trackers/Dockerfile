FROM python:3.9-bullseye as similari-base
WORKDIR /opt
RUN git clone https://github.com/insight-platform/Similari.git --branch=v0.22.9 \
    && cd Similari/docker/common/ \
    && bash install-basic-deps.sh

FROM similari-base as chef
ENV PATH="/root/.cargo/bin:$PATH"
RUN rustc -V

FROM chef AS planner
WORKDIR /opt/Similari
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
WORKDIR /opt/Similari
COPY --from=planner /opt/Similari/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY --from=similari-base /opt/Similari .
ENV RUSTFLAGS=" -C target-cpu=native -C opt-level=3"
RUN maturin build --release --out dist

FROM python:3.9-bullseye as base
COPY --from=builder /opt/Similari/dist/ /opt/dist/
COPY requirements/trackers/requirements.txt .
RUN apt-get update \
    && apt-get install -y python3-opencv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install /opt/dist/* \
    && python3 -m pip install -r requirements.txt
RUN git clone https://github.com/JonathonLuiten/TrackEval.git /tmp/TrackEval \
    && cd /tmp/TrackEval \
    && git checkout b96772a1298f054038100b7de795726fcc96c7fb \
    && python3 setup.py install \
    && rm -rf /tmp/TrackEval
WORKDIR /opt/app
COPY src/trackers trackers
COPY src/dataset dataset
ENTRYPOINT ["python"]
CMD []