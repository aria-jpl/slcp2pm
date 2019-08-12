#!/usr/bin/env python
from __future__ import print_function

import os, sys, requests, json
from pprint import pprint
from subprocess import check_output
CONF_FILE = "/home/ops/slcp2pm/conf/settings.conf"


def get_version():
    """Get dataset version."""

    DS_VERS_CFG = os.path.normpath(
                      os.path.join(
                          os.path.dirname(os.path.abspath(__file__)),
                          '..', 'conf', 'dataset_versions.json'))
    with open(DS_VERS_CFG) as f:
        ds_vers = json.load(f)
    return ds_vers['S1-LAR']


def check_lar(es_url, es_index, id):
    """Query for log_amp_ratio with specified input ID."""

    query = {
        "query":{
            "bool":{
                "must":[
                    {"term":{"_id":id}},
                ]
            }
        },
        "fields": [],
    }

    if es_url.endswith('/'):
        search_url = '%s%s/_search' % (es_url, es_index)
    else:
        search_url = '%s/%s/_search' % (es_url, es_index)
    r = requests.post(search_url, data=json.dumps(query))
    if r.status_code != 200:
        print("Failed to query %s:\n%s" % (es_url, r.text), file=sys.stderr)
        print("query: %s" % json.dumps(query, indent=2), file=sys.stderr)
        print("returned: %s" % r.text, file=sys.stderr)
    if r.status_code == 404: return 0, 'NONE'
    else: r.raise_for_status()
    result = r.json()
    pprint(result)
    total = result['hits']['total']
    if total == 0: id = 'NONE'
    else: id = result['hits']['hits'][0]['_id']
    return total, id


if __name__ == "__main__":
    # uu = UrlUtils()
    # es_url = uu.rest_url
    es_url = check_output(['grep GRQ_URL= {} | cut -d= -f2'.format(CONF_FILE)], shell=True).decode("utf-8")
    es_index = 'grq_%s_s1-lar' % (get_version())
    total, id = check_lar(es_url, es_index, sys.argv[1])
    if total > 0:
        with open('lar_found.txt', 'w') as f:
            f.write("%s\n" % id)
