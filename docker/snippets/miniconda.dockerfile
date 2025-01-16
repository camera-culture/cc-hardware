# syntax = devthefuture/dockerfile-x
# Install miniconda

RUN apt-get update && \
        apt-get install --no-install-recommends -y \
            wget && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ARG CONDA_PATH=${USERHOME}/conda
RUN ARCH=$(uname -m) && \
      if [ "${ARCH}" = "x86_64" ]; then \
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
      elif [ "${ARCH}" = "aarch64" ]; then \
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
      elif [ "${ARCH}" = "ppc64le" ]; then \
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh"; \
      else \
            echo "Unsupported architecture: ${ARCH}"; exit 1; \
      fi && \
      mkdir -p ${CONDA_PATH} && \
      wget ${MINICONDA_URL} -O ${CONDA_PATH}/miniconda.sh && \
      bash ${CONDA_PATH}/miniconda.sh -b -u -p ${CONDA_PATH} && \
      rm -rf ${CONDA_PATH}/miniconda.sh

ENV PATH "${CONDA_PATH}/bin:${PATH}"
ARG PATH "${CONDA_PATH}/bin:${PATH}"
ENV CONDA_ALWAYS_YES "true"

RUN conda create -n ${PROJECT} -y && \
      echo ". '${CONDA_PATH}/etc/profile.d/conda.sh'" >> ${USERSHELLPROFILE} && \
      echo "conda activate ${PROJECT}" >> ${USERSHELLPROFILE}
ARG CONDA_CHANNELS=""
RUN [ -z "${CONDA_CHANNELS}" ] || \
      for channel in ${CONDA_CHANNELS}; do conda config --add channels ${channel}; done

ARG CONDA_PACKAGES=""
RUN [ -z "${CONDA_PACKAGES}" ] || conda install ${CONDA_PACKAGES}
