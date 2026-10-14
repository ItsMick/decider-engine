# Build natively with `docker buildx build --platform linux/arm64`.
# Amazon Linux 2023 ensures the bootstrap and runtime use a compatible glibc.
FROM public.ecr.aws/amazonlinux/amazonlinux:2023 AS build
RUN dnf install -y gcc gcc-c++ make perl openssl-devel curl tar gzip && dnf clean all
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.98.0
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY lambda/Cargo.toml lambda/Cargo.lock ./lambda/
COPY lambda/src ./lambda/src
ENV RUSTFLAGS="-C target-cpu=neoverse-n1"
RUN test "$(uname -m)" = aarch64 && cargo build --manifest-path lambda/Cargo.toml --release --locked

FROM public.ecr.aws/lambda/provided:al2023
COPY --from=build /build/lambda/target/release/decider-lambda /var/runtime/bootstrap
# Run the pinned hash-verifying downloader BEFORE building. Licenses and provenance
# travel with the weights. No model network access or quantization per invocation.
COPY models/v21 /opt/model
ENV DECIDER_MODEL_DIR=/opt/model DECIDER_PRECISION=checkpoint RAYON_NUM_THREADS=4 DECIDER_REQUIRE_SDOT=1
ENTRYPOINT ["/var/runtime/bootstrap"]
