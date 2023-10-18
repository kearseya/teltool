FROM python:3.10

USER root

COPY . /teltool

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN curl -L https://github.com/lh3/minimap2/releases/download/v2.26/minimap2-2.26_x64-linux.tar.bz2 | tar -jxvf -; mv ./minimap2-2.26_x64-linux/minimap2 /opt/venv/bin
RUN pip3 install --upgrade pip3; pip3 install Cython pysam==0.21.0 numpy scikit-learn; pip3 install -r /teltool/requirements.txt; pip3 install /teltool

CMD ["/bin/sh"]


