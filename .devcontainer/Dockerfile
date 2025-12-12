# Use R image based on Ubuntu instead of Debian sid
FROM rocker/r-ver:4.3.2

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CRAN_MIRROR=https://cran.r-project.org

# Update system packages and install essential tools
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    gfortran \
    # System libraries needed for R packages
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libcairo2-dev \
    libxt-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libfontconfig1-dev \
    # Database drivers
    libpq-dev \
    libmariadb-dev \
    unixodbc-dev \
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
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Create a virtual environment for Python packages
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Activate virtual environment and install Python packages for data science
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
    "numpy<2.0" \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    notebook \
    jupyterlab \
    tensorflow \
    keras \
    scipy \
    plotly \
    dash \
    streamlit \
    requests \
    beautifulsoup4 \
    openpyxl \
    xlrd

# Install R packages using install.packages first (pacman dependency)
RUN R -e "install.packages('pacman', repos='https://cran.r-project.org')"

# Set working directory
WORKDIR /workspace

# Copy package installation script
COPY install-packages.sh /install-packages.sh
RUN chmod +x /install-packages.sh

# Install R packages
RUN /install-packages.sh

RUN apt update && apt install -y pandoc libglpk40 libzmq5

# Set default command
CMD ["/bin/bash"]