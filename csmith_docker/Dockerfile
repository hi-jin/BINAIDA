# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV INSTALL_PREFIX=/usr/local
ENV PATH="$INSTALL_PREFIX/bin:$PATH"

# Install required packages
RUN apt-get update && \
    apt-get install -y \
    g++ \
    cmake \
    m4 \
    llvm \
    clang \
    curl \
    git \
    socat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone and install Csmith
RUN git clone https://github.com/csmith-project/csmith.git /opt/csmith && \
    cd /opt/csmith && \
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX . && \
    make && \
    make install

# Copy the script to handle requests
COPY handle_request.sh /usr/local/bin/handle_request.sh
RUN chmod +x /usr/local/bin/handle_request.sh

# Expose port 8080
EXPOSE 8080

# Start a simple HTTP server to handle requests
CMD ["socat", "TCP-LISTEN:8080,fork", "EXEC:/usr/local/bin/handle_request.sh"]
