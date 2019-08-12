FROM hysds/pge-isce_giant:latest

LABEL description="PGE container for LAR product generation"

# create work directory
RUN set -ex \
 && mkdir -p /data/work \
 && chmod -R 755 /data \
 && chown -R ops:ops /data

USER ops

# copy code ensure proper permissions, and move dependencies to final locations
COPY . /home/ops/verdi/ops/slcp2pm
RUN set -ex \
 && sudo chown -R ops:ops /home/ops/verdi/ops/slcp2pm \
 && cd /home/ops/verdi/ops/slcp2pm/src \
 && ./compile.sh

# set entrypoint
WORKDIR /home/ops
CMD ["/bin/bash", "--login"]
