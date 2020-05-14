FROM hysds/isce2:latest

LABEL description="PGE container for LAR product generation"

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
