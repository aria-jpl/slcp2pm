#!/bin/bash
BASE_PATH=$(dirname "${BASH_SOURCE}")
BASE_PATH=$(cd "${BASE_PATH}"; pwd)
SLCP_PROD=$1
SWATH=$2

echo "##########################################" 1>&2
echo -n "Running S1 log amp ratio generation: " 1>&2
date 1>&2
source /opt/isce2/isce_env.sh && python3 $BASE_PATH/create_lar.py $SLCP_PROD > create_lar.log 2>&1
STATUS=$?
echo -n "Finished running S1 log amp ratio generation: " 1>&2
date 1>&2
if [ $STATUS -ne 0 ]; then
  echo "Failed to run S1 log amp ratio generation." 1>&2
  cat create_lar.log 1>&2
  echo "{}"
  exit $STATUS
fi
