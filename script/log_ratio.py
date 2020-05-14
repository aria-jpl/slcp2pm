#!/usr/bin/env python3
from __future__ import division
from builtins import range
from past.utils import old_div
import os
import sys
import glob
import shutil
import ntpath
import pickle
import datetime
import argparse
import numpy as np
import numpy.matlib
from xml.etree.ElementTree import ElementTree
from subprocess import check_call

import isce
import isceobj
from imageMath import IML


SCR_PATH = os.path.abspath(os.path.dirname(__file__))
BIN_PATH = os.path.join(os.path.dirname(SCR_PATH), "src")


def runCmd(cmd):
    print("{}".format(cmd))
    #status = os.system(cmd)
    status = check_call(cmd, shell=True)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def getWidth(xmlfile):
    xmlfp = None
    try:
        xmlfp = open(xmlfile,'r')
        print('reading file width from: {0}'.format(xmlfile))
        xmlx = ElementTree(file=xmlfp).getroot()
        tmp = xmlx.find("component[@name='coordinate1']/property[@name='size']/value")
        if tmp == None:
            tmp = xmlx.find("component[@name='Coordinate1']/property[@name='size']/value")
        width = int(tmp.text)
        print("file width: {0}".format(width))
    except (IOError, OSError) as strerr:
        print("IOError: %s" % strerr)
        return []
    finally:
        if xmlfp is not None:
            xmlfp.close()
    return width


def getLength(xmlfile):
    xmlfp = None
    try:
        xmlfp = open(xmlfile,'r')
        print('reading file length from: {0}'.format(xmlfile))
        xmlx = ElementTree(file=xmlfp).getroot()
        tmp = xmlx.find("component[@name='coordinate2']/property[@name='size']/value")
        if tmp == None:
            tmp = xmlx.find("component[@name='Coordinate2']/property[@name='size']/value")
        length = int(tmp.text)
        print("file length: {0}".format(length))
    except (IOError, OSError) as strerr:
        print("IOError: %s" % strerr)
        return []
    finally:
        if xmlfp is not None:
            xmlfp.close()
    return length


def create_xml(fileName, width, length, fileType):
    if fileType == 'slc':
        image = isceobj.createSlcImage()
    elif fileType == 'int':
        image = isceobj.createIntImage()
    elif fileType == 'amp':
        image = isceobj.createAmpImage()
    elif fileType == 'rmg':
        image = isceobj.Image.createUnwImage()
    elif fileType == 'float':
        image = isceobj.createImage()
        image.setDataType('FLOAT')

    image.setFilename(fileName)
    image.setWidth(width)
    image.setLength(length)
        
    image.setAccessMode('read')
    #image.createImage()
    image.renderVRT()
    image.renderHdr()
    #image.finalizeImage()


def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser( description='log ratio')
    parser.add_argument('-mdir', dest='mdir', type=str, required=True, help='master directory containing the bursts')
    parser.add_argument('-sdir', dest='sdir', type=str, required=True, help='slave directory containing the bursts')
    parser.add_argument('-gdir', dest='gdir', type=str, required=True,
                        help='geometric directory containing the lat/lon files')
    parser.add_argument('-rlks', dest='rlks', type=int, default=0, help='number of range looks')
    parser.add_argument('-alks', dest='alks', type=int, default=0, help='number of azimuth looks')
    parser.add_argument('-ssize', dest='ssize', type=float, default=1.0,
                        help='output geocoded sample size. default: 1.0 arcsec')
    return parser.parse_args()


if __name__ == '__main__':
    SCR_DIR = SCR_PATH

    inps = cmdLineParse()

    mbursts = sorted(glob.glob(os.path.join(inps.mdir, 'burst_*.slc')))
    sbursts = sorted(glob.glob(os.path.join(inps.sdir, 'burst_*.slc')))

    nmb = len(mbursts) #number of master bursts
    nsb = len(sbursts) #number of slave bursts

    lats = sorted(glob.glob(os.path.join(inps.gdir, 'lat_*.rdr')))
    lons = sorted(glob.glob(os.path.join(inps.gdir, 'lon_*.rdr')))

    nb = nmb

    for i in range(nb):
        print('+++++++++++++++++++++++++++++++++++')
        print('processing burst {} of {}'.format(i+1, nb))
        print('+++++++++++++++++++++++++++++++++++')

        # find slave burst here
        master_burst = ntpath.basename(mbursts[i])
        slave_burst_id = -1
        for ii in range(nsb):
            slave_burst = ntpath.basename(sbursts[ii])
            if slave_burst == master_burst:
                slave_burst_id = ii
                break
        if slave_burst_id == -1:
            print('no matching slave burst found, skip this burst')
            continue

        amp = 'amp_%02d.amp' % (i+1)
        # cmd = "imageMath.py -e='(real(a)!=0)*(real(b)!=0)*(imag(a)!=0)*(imag(b)!=0)*sqrt(real(a)*real(a)+imag(a)*imag(a));(real(a)!=0)*(real(b)!=0)*(imag(a)!=0)*(imag(b)!=0)*sqrt(real(b)*real(b)+imag(b)*imag(b))' --a={} --b={} -o {} -t float -s BIP".format(
        #     mbursts[i],
        #     sbursts[i],
        #     amp)
        # runCmd(cmd)

        width = getWidth(mbursts[i] + '.xml')
        length = getLength(mbursts[i] + '.xml')

        width_looked = int(old_div(width,inps.rlks))
        length_looked = int(old_div(length,inps.alks))

        master = np.fromfile(mbursts[i], dtype=np.complex64).reshape(length, width)
        slave = np.fromfile(sbursts[slave_burst_id], dtype=np.complex64).reshape(length, width)

        amp_data = np.zeros((length, width*2), dtype=np.float)
        amp_data[:, 0:width * 2:2] = np.absolute(master) * (np.absolute(slave) != 0)
        amp_data[:, 1:width * 2:2] = np.absolute(slave) * (np.absolute(master) != 0)
        amp_data.astype(np.float32).tofile(amp)
        create_xml(amp, width, length, 'amp')

        amp_looked = 'amp_%02d_%drlks_%dalks.amp' % (i + 1, inps.rlks, inps.alks)
        cmd = "{}/look.py -i {} -o {} -r {} -a {}".format(SCR_DIR, amp,  amp_looked, inps.rlks, inps.alks)
        runCmd(cmd)

        # mburst_looked = 'master_%02d_%drlks_%dalks.slc' % (i+1,inps.rlks,inps.alks)
        # cmd = "look.py -i {} -o {} -r {} -a {}".format(
        #     mbursts[i], 
        #     mburst_looked,
        #     inps.rlks,
        #     inps.alks)
        # runCmd(cmd)

        # sburst_looked = 'slave_%02d_%drlks_%dalks.slc' % (i+1,inps.rlks,inps.alks)
        # cmd = "look.py -i {} -o {} -r {} -a {}".format(
        #     sbursts[i], 
        #     sburst_looked,
        #     inps.rlks,
        #     inps.alks)
        # runCmd(cmd)

        lat_looked = 'lat_%02d_%drlks_%dalks.rdr' % (i + 1, inps.rlks, inps.alks)
        #lat = os.path.join(inps.gdir, 'lat_%02d.rdr'%(i+1))
        cmd = "{}/look.py -i {} -o {} -r {} -a {}".format(SCR_DIR,
            lats[slave_burst_id], 
            lat_looked,
            inps.rlks,
            inps.alks)
        runCmd(cmd)

        lon_looked = 'lon_%02d_%drlks_%dalks.rdr' % (i + 1, inps.rlks, inps.alks)
        #lon = os.path.join(inps.gdir, 'lon_%02d.rdr'%(i+1))
        cmd = "{}/look.py -i {} -o {} -r {} -a {}".format(
            SCR_DIR,
            lons[slave_burst_id], 
            lon_looked,
            inps.rlks,
            inps.alks)
        runCmd(cmd)

        logr_looked = 'logr_%02d_%drlks_%dalks.float' % (i + 1, inps.rlks, inps.alks)
        # cmd = "imageMath.py -e='log10((a_0)/(a_1+(a_1==0)))*(a_0!=0)*(a_1!=0)' --a={} -o {} -t float -s BIP".format(
        #     amp_looked, 
        #     logr_looked)
        # runCmd(cmd)

        amp_looked_data = np.fromfile(amp_looked, dtype=np.float32).reshape(length_looked, width_looked * 2)
        m = amp_looked_data[:, 0:width_looked * 2:2]
        s = amp_looked_data[:, 1:width_looked * 2:2]
        # Only for S1-LAR before v2.0! Apre/Aco:
        # logr_looked_data = np.log10(     (m+(m==0))     /    (s+(s==0))        ) * (m!=0) * (s!=0)

        # Only for S1-LAR v2.0 onwards! Aco/Apre (-ve value is openwater flood, +ve value is veg-flood)
        logr_looked_data = np.log10(old_div((s + (s == 0)), (m + (m == 0)))) * (m != 0) * (s != 0)

        #remove white edges
        upper_edge = 0
        for k in range(length_looked):
            if logr_looked_data[k, int(old_div(width_looked, 2))] != 0:
                upper_edge = k
                break

        lower_edge = length_looked - 1
        for k in range(length_looked):
            if logr_looked_data[length_looked - 1 - k, int(old_div(width_looked, 2))] != 0:
                lower_edge = length_looked - 1 - k
                break

        left_edge = 0
        for k in range(width_looked):
            if logr_looked_data[int(old_div(length_looked, 2)), k] != 0:
                left_edge = k
                break

        right_edge = width_looked-1
        for k in range(width_looked):
            if logr_looked_data[int(old_div(length_looked, 2)), width_looked - 1 - k] != 0:
                right_edge = width_looked-1-k
                break

        print('four edgeds: lower: {}, upper: {}, left: {}, right: {}'.format(lower_edge, upper_edge, left_edge, right_edge))
        flag = np.zeros((length_looked, width_looked), dtype=np.float)
        delta = 3
        flag[upper_edge + delta:lower_edge - delta, left_edge + delta:right_edge - delta] = 1.0
        logr_looked_data *= flag

        logr_looked_data.astype(np.float32).tofile(logr_looked)
        create_xml(logr_looked, width_looked, length_looked, 'float')

        #width = getWidth(lon_looked + '.xml')
        #length = getLength(lon_looked + '.xml')
        lat_looked_data = np.fromfile(lat_looked, dtype=np.float64).reshape(length_looked, width_looked)
        lon_looked_data = np.fromfile(lon_looked, dtype=np.float64).reshape(length_looked, width_looked)

        lat_max = np.amax(lat_looked_data)
        lat_min = np.amin(lat_looked_data)
        lon_max = np.amax(lon_looked_data)
        lon_min = np.amin(lon_looked_data)
        bbox = "{}/{}/{}/{}".format(lat_min, lat_max, lon_min, lon_max)

        logr_looked_geo = 'logr_%02d_%drlks_%dalks.float.geo' % (i+1,inps.rlks,inps.alks)
        cmd = "{}/geo_with_ll.py -input {} -output {} -lat {} -lon {} -bbox={} -ssize {} -rmethod {}".format(
            SCR_DIR,
            logr_looked, 
            logr_looked_geo,
            lat_looked,
            lon_looked,
            bbox,
            1.0,
            1)
        runCmd(cmd)

        amp_looked_geo = 'amp_%02d_%drlks_%dalks.amp.geo' % (i+1,inps.rlks,inps.alks)
        cmd = "{}/geo_with_ll.py -input {} -output {} -lat {} -lon {} -bbox={} -ssize {} -rmethod {}".format(
            SCR_DIR,
            amp_looked,
            amp_looked_geo,
            lat_looked,
            lon_looked,
            bbox,
            inps.ssize,
            1)
        runCmd(cmd)

        os.remove(amp)
        os.remove(amp_looked)
        os.remove(lat_looked)
        os.remove(lon_looked)
        os.remove(logr_looked)

        os.remove(amp+'.xml')
        os.remove(amp_looked+'.xml')
        os.remove(lat_looked+'.xml')
        os.remove(lon_looked+'.xml')
        os.remove(logr_looked+'.xml')

        os.remove(amp+'.vrt')
        os.remove(amp_looked+'.vrt')
        os.remove(lat_looked+'.vrt')
        os.remove(lon_looked+'.vrt')
        os.remove(logr_looked+'.vrt')

#log_ratio.py -mdir /u/hm/NC/data/S1-COH_STCM3S3_TN077_20160929T231332-20161011T231433_s1-resorb-v1.0/master -sdir /u/hm/NC/data/S1-COH_STCM3S3_TN077_20160929T231332-20161011T231433_s1-resorb-v1.0/fine_coreg -gdir /u/hm/NC/data/S1-COH_STCM3S3_TN077_20160929T231332-20161011T231433_s1-resorb-v1.0/geom_master -rlks 7 -alks 2
