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
    # Python and pip
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Additional tools
    wget \
    curl \
    vim \
    nano \
    htop \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Create a virtual environment for Python packages
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir \
    "numpy<2.0" \
    pandas \
    scikit-learn \
    matplotlib \
    scipy \
    requests \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# Enable binary packages for faster installation
RUN echo 'options(pkgType = "binary", repos = c(CRAN = "https://cran.r-project.org"))' >> /usr/local/lib/R/etc/Rprofile.site

# Install pacman first for R package management
RUN R --vanilla -e "install.packages('pacman')"

# Install BiocManager and Bioconductor packages
RUN R --vanilla -e "install.packages('BiocManager')" && \
    R --vanilla -e "BiocManager::install('EBImage', ask=FALSE)"

# Install core required packages only
RUN R --vanilla -e "\
pacman::p_load( \
  targets, tarchetypes, \
  tidyr, jsonlite, reticulate, \
  repos='https://cran.r-project.org', \
  install=TRUE, update=FALSE, character.only=TRUE)"

# Set working directory
WORKDIR /workspace

# Create startup script that installs missing packages dynamically
COPY <<EOF /opt/init.sh
#!/bin/bash
# This script installs any missing packages without rebuilding the image
echo "Checking for missing R packages..."
R --vanilla -e "
pacman::p_load(targets, tarchetypes, tidyr, jsonlite, reticulate, 
               quiet=TRUE, warn.conflicts=FALSE, install=TRUE, update=FALSE, character.only=TRUE)
cat('✓ Core packages ready\n')
"
exec "$@"
EOF

RUN chmod +x /opt/init.sh

# Pre-initialize PyTorch with R reticulate
RUN R --vanilla -e "library(reticulate); use_python('/opt/venv/bin/python'); import('torch'); cat('✓ PyTorch ready\n')"

# Set entrypoint to startup script
ENTRYPOINT ["/opt/init.sh"]
CMD ["/bin/bash"]