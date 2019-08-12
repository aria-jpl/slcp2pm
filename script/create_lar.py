#!/usr/bin/env python3 
import os, sys, re, json, shutil, traceback, logging
from subprocess import check_call, check_output
from datetime import datetime
from check_lar import check_lar, get_version


log_format = "[%(asctime)s: %(levelname)s/%(funcName)s] %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger('create_lar')


BASE_PATH = os.path.dirname(__file__)

CONF_FILE = "/home/ops/verdi/ops/slcp2pm/conf/settings.conf"
MISSION_RE = re.compile(r'^(S1\w)_')
SLCP_RE = re.compile(r'S1-SLCP_(.+_s(\d)-.+?)-(v.+)$')
RLKS = 7
ALKS = 2


def create_met_json(id, version, slcp_met_file, met_file):
    """Write met json."""

    # get metadata
    with open(slcp_met_file) as f:
        md = json.load(f)

    # overwrite
    md['dataset_type'] = "log_amp_ratio"
    md['product_type'] = "log_amp_ratio"
    md['archive_filename'] = id
    md['range_looks'] = RLKS
    md['azimuth_looks'] = ALKS


    # write out met json
    with open(met_file, 'w') as f:
        json.dump(md, f, indent=2, sort_keys=True)


def create_dataset_json(id, version, slcp_ds_file, ds_file):
    """Write dataset json."""

    # get metadata
    with open(slcp_ds_file) as f:
        ds = json.load(f)

    # overwrite
    ds['creation_timestamp'] = "%sZ" % datetime.utcnow().isoformat()
    ds['version'] = version
    ds['label'] = id

    # write out dataset json
    with open(ds_file, 'w') as f:
        json.dump(ds, f, indent=2)


def lar_exists(es_url, es_index, id):
    """Check if log amp ratio product exists in GRQ."""

    total, id = check_lar(es_url, es_index, id)
    if total > 0: return True
    return False


def call_noerr(cmd):
    """Run command and warn if exit status is not 0."""

    try: check_call(cmd, shell=True)
    except Exception as e:
        logger.warn("Got exception running {}: {}".format(cmd, str(e)))
        logger.warn("Traceback: {}".format(traceback.format_exc()))


def main(slcp_dir):
    """HySDS PGE wrapper for log_amp_ratio generation."""

    # save cwd (working directory)
    cwd = os.getcwd()

    # extract info from SLCP product
    slcp_dir = os.path.abspath(os.path.normpath(slcp_dir))
    slcp_id = os.path.basename(slcp_dir)
    slcp_met_file = os.path.join(slcp_dir, "{}.met.json".format(slcp_id))
    slcp_ds_file = os.path.join(slcp_dir, "{}.dataset.json".format(slcp_id))

    # get dataset version, set dataset ID and met/dataset JSON files
    match = SLCP_RE.search(slcp_id)
    if not match:
        raise(RuntimeError("Failed to recognize SLCP id: {}".format(slcp_id)))
    id_base = "S1-LAR_{}".format(match.group(1))
    swath = match.group(2)
    slcp_version = match.group(3)
    version = get_version()
    id = "{}-{}".format(id_base, version)
    prod_dir = os.path.abspath(id)
    met_file = os.path.join(prod_dir, "{}.met.json".format(id))
    ds_file = os.path.join(prod_dir, "{}.dataset.json".format(id))

    # get endpoint configurations
    # uu = UrlUtils()
    # es_url = uu.rest_url
    # es_index = "{}_{}_s1-lar".format(uu.grq_index_prefix, version)
    es_url = check_output(['grep GRQ_URL= {} | cut -d= -f2'.format(CONF_FILE)], shell=True).decode("utf-8")
    es_index = "grq_{}_s1-lar".format(version)

    # check if log_amp_ratio already exists
    logger.info("GRQ url: {}".format(es_url))
    logger.info("GRQ index: {}".format(es_index))
    logger.info("Product ID for version {}: {}".format(version, id))
    if lar_exists(es_url, es_index, id):
        logger.info("{} log_amp_ratio for {}".format(version, id_base) +
                    " was previously generated and exists in GRQ database.")

        # cleanup SLCP dir
        logger.info("Removing {}.".format(slcp_dir))
        try: shutil.rmtree(slcp_dir)
        except: pass
        return 0

    # generate log amp ratio
    lar_cmd = [ "{}/slcp2lar_S1.sh".format(BASE_PATH), slcp_dir, swath, str(RLKS), str(ALKS)]
    lar_cmd_line = " ".join(lar_cmd)
    logger.info("Calling slcp2lar_S1.sh: {}".format(lar_cmd_line))
    check_call(lar_cmd_line, shell=True)
        
    # create product directory
    os.makedirs(prod_dir, 0o755)

    # move all products
    call_noerr("mv -f s{}/* {}/".format(swath, prod_dir))

    # generate met and dataset JSON
    create_met_json(id, version, slcp_met_file, met_file)
    create_dataset_json(id, version, slcp_ds_file, ds_file)
    
    # clean out SLCP prod
    try: shutil.rmtree(slcp_dir)
    except: pass


if __name__ == '__main__':
    try: status = main(sys.argv[1])
    except Exception as e:
        with open('_alt_error.txt', 'w') as f:
            f.write("%s\n" % str(e))
        with open('_alt_traceback.txt', 'w') as f:
            f.write("%s\n" % traceback.format_exc())
        raise
    sys.exit(status)
