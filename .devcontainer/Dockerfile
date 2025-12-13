FROM devxygmbh/r-ubuntu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update system packages and install essential tools
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    gfortran \
    # System libraries needed for R packages
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libcairo2-dev \
    libxt-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libfftw3-dev \
    # Version control
    git \
    # Additional tools
    wget \
    curl \
    vim \
    nano \
    htop \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libglpk40 \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages('targets')"

WORKDIR /workspace

CMD ["/bin/bash"]
