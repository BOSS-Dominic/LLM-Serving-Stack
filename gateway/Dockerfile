# Multi-stage build: the runtime image contains a static binary, no toolchain.
FROM golang:1.26-alpine AS build

WORKDIR /src

# Cache the module layer separately from the source.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ARG VERSION=dev
RUN CGO_ENABLED=0 GOOS=linux go build \
    -trimpath \
    -ldflags "-s -w -X main.version=${VERSION}" \
    -o /out/gateway .

FROM alpine:3.20

# ca-certificates for TLS backends, wget for the container healthcheck.
RUN apk add --no-cache ca-certificates wget \
    && adduser -D -H -u 10001 gateway

COPY --from=build /out/gateway /usr/local/bin/gateway
COPY config.yaml /etc/gateway/config.yaml

USER gateway
EXPOSE 9000

ENTRYPOINT ["/usr/local/bin/gateway"]
CMD ["-config", "/etc/gateway/config.yaml"]
