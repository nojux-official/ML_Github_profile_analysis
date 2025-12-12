# Use R 4.5 image based on Ubuntu
FROM rocker/r-ver:4.5

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CRAN_MIRROR=https://cran.r-project.org

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

# Enable binary packages for faster installation
RUN echo 'options(pkgType = "binary", repos = c(CRAN = "https://cran.r-project.org"))' >> /usr/local/lib/R/etc/Rprofile.site

# Install pacman first for R package management
RUN R --vanilla -e "options(repos='https://cran.r-project.org'); install.packages('pacman')"

# Install BiocManager and Bioconductor packages
RUN R --vanilla -e "options(repos='https://cran.r-project.org'); install.packages('BiocManager')" && \
    R --vanilla -e "BiocManager::install('EBImage', ask=FALSE)"

# Install core required packages
RUN R --vanilla -e "\
pacman::p_load( \
  targets, tarchetypes, \
  tidyr, jsonlite, \
  torch, luz, torchvision, \
  repos='https://cran.r-project.org', \
  install=TRUE, update=FALSE, character.only=TRUE)"

# Set working directory
WORKDIR /workspace

RUN R --vanilla -e "library(torch); cat('✓ Torch R package ready\n')"

# Set entrypoint to startup script
CMD ["/bin/bash"]
