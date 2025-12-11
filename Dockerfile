# use miniforge base
FROM condaforge/miniforge3:23.11.0-0

USER root

# Set timezone non-interactively and install packages
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    texlive-luatex \
    texlive-latex-extra && \
    rm -rf /var/lib/apt/lists/*

# copy lockfile
COPY conda-lock.yml /tmp/conda-lock.yml

# install conda-lock and all packages in dockerlock environment
RUN conda install -n base -c conda-forge conda-lock -y \
    && conda-lock install -n dockerlock /tmp/conda-lock.yml \
    && conda clean --all -y -f

# make the dockerlock environment global
ENV PATH="/opt/conda/envs/dockerlock/bin:/usr/bin:$PATH"

# use login shell to pick up PATH
SHELL ["/bin/bash", "-l", "-c"]

RUN /opt/conda/envs/dockerlock/bin/pip install \
    ucimlrepo \
    "deepchecks[tabular]" \
    anywidget \
    "vegafusion[embed]>=2.0.0" \
    "altair_ally>=0.1.1" \
    "vl-convert-python>=1.8.0" \
    pyarrow \
    quarto-cli

# Register the Jupyter kernel for Quarto to use
RUN /opt/conda/envs/dockerlock/bin/python -m ipykernel install \
    --name dockerlock \
    --display-name "Python (dockerlock)"


# working directory
WORKDIR /workspace

# expose port
EXPOSE 8888

# run JupyterLab by default
CMD ["/opt/conda/envs/dockerlock/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]