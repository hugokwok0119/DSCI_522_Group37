# use miniforge base
FROM condaforge/miniforge3:23.11.0-0

USER root

# Set timezone non-interactively and install packages
# build-essential INCLUDES 'make', so this installs it correctly.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    texlive-luatex \
    texlive-latex-extra && \
    rm -rf /var/lib/apt/lists/*

# Install quarto -----
ARG TARGETARCH
ARG QUARTO_VERSION=1.8.26
RUN if [ "$TARGETARCH" = "amd64" ]; then \
  QUARTO_ARCH="amd64"; \
  elif [ "$TARGETARCH" = "arm64" ]; then \
  QUARTO_ARCH="arm64"; \
  else \
  echo "Unsupported architecture: $TARGETARCH" && exit 1; \
  fi && \
  curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${QUARTO_ARCH}.tar.gz && \
  mkdir -p /opt/quarto && \
  tar -xzf quarto-${QUARTO_VERSION}-linux-${QUARTO_ARCH}.tar.gz -C /opt/quarto --strip-components=1 && \
  rm quarto-${QUARTO_VERSION}-linux-${QUARTO_ARCH}.tar.gz && \
  ln -s /opt/quarto/bin/quarto /usr/local/bin/quarto

# copy lockfile
COPY conda-lock.yml /tmp/conda-lock.yml

# install conda-lock and packages
RUN conda install -n base -c conda-forge conda-lock -y \
    && conda-lock install -n MDS_Group37 /tmp/conda-lock.yml \
    && conda clean --all -y -f

# make the environment global
ENV PATH="/opt/conda/envs/MDS_Group37/bin:/usr/bin:$PATH"

# use login shell to pick up PATH
SHELL ["/bin/bash", "-l", "-c"]

# Install pip packages into the correct environment
RUN /opt/conda/envs/MDS_Group37/bin/pip install \
    ucimlrepo \
    "deepchecks[tabular]" \
    anywidget \
    "vegafusion[embed]>=2.0.0" \
    "altair_ally>=0.1.1" \
    "vl-convert-python>=1.8.0" \
    pyarrow

# Register the Jupyter kernel
RUN /opt/conda/envs/MDS_Group37/bin/python -m ipykernel install \
    --name MDS_Group37 \
    --display-name "Python (MDS_Group37)"

# working directory
WORKDIR /workspace

# expose port
EXPOSE 8888

# run JupyterLab by default
# Updated path to match the new env name
CMD ["/opt/conda/envs/MDS_Group37/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]