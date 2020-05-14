#!/usr/bin/env python3

#Cunren Liang, JPL/Caltech


from __future__ import division
from past.utils import old_div
import os
import sys
import argparse
from xml.etree.ElementTree import ElementTree
from subprocess import check_call

import isce
import isceobj


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

def ampLooks(inps):


    inWidth = getWidth(inps.input + '.xml')
    inLength = getLength(inps.input + '.xml')
    outWidth = int(old_div(inWidth,inps.rlks))
    outLength = int(old_div(inLength,inps.alks))

    #run it
    #cmd = 'echo -e "{}\n{}\n{} {}\n{} {}\n" | {}/rilooks'.format(inps.input, inps.output, inWidth, inLength, inps.rlks, inps.alks, BIN_PATH)
    #it seems that echo does not require -e in this situation, strange
    cmd = 'echo "{}\n{}\n{} {}\n{} {}\n" | {}/rilooks'.format(inps.input, inps.output, inWidth, inLength, inps.rlks, inps.alks, BIN_PATH)
    runCmd(cmd)

    #get xml file for amplitude image
    ampImage = isceobj.createAmpImage()
    ampImage.setFilename(inps.output)
    ampImage.setWidth(outWidth)
    ampImage.setLength(outLength)
    ampImage.setAccessMode('read')
    #ampImage.createImage()
    ampImage.renderVRT()
    ampImage.renderHdr()
    #ampImage.finalizeImage()

def intLooks(inps):

    inWidth = getWidth(inps.input + '.xml')
    inLength = getLength(inps.input + '.xml')
    outWidth = int(old_div(inWidth,inps.rlks))
    outLength = int(old_div(inLength,inps.alks))

    #run program here
    cmd = 'echo "{}\n{}\n{} {}\n{} {}\n" | {}/cpxlooks'.format(inps.input, inps.output, inWidth, inLength, inps.rlks, inps.alks, BIN_PATH)
    runCmd(cmd)
    
    #get xml file for interferogram
    intImage = isceobj.createIntImage()
    intImage.setFilename(inps.output)
    intImage.setWidth(outWidth)
    intImage.setLength(outLength)
    intImage.setAccessMode('read')
    #intImage.createImage()
    intImage.renderVRT()
    intImage.renderHdr()
    #intImage.finalizeImage()


def mskLooks(inps):

    inWidth = getWidth(inps.input + '.xml')
    inLength = getLength(inps.input + '.xml')
    outWidth = int(old_div(inWidth,inps.rlks))
    outLength = int(old_div(inLength,inps.alks))

    #look_msk infile outfile nrg nrlks nalks
    #run program here
    cmd = '{}/look_msk {} {} {} {} {}'.format(BIN_PATH, inps.input, inps.output, inWidth, inps.rlks, inps.alks)
    runCmd(cmd)
    
    #get xml file for interferogram
    image = isceobj.createImage()
    accessMode = 'read'
    dataType = 'BYTE'
    bands = 1
    scheme = 'BIL'
    width = outWidth
    image.initImage(inps.output, accessMode, width, dataType, bands=bands, scheme=scheme)
    descr = 'Radar shadow-layover mask. 1 - Radar Shadow. 2 - Radar Layover. 3 - Both.'
    image.addDescription(descr)
    image.renderVRT()
    image.renderHdr()
    #image.finalizeImage()


def hgtLooks(inps):

    inWidth = getWidth(inps.input + '.xml')
    inLength = getLength(inps.input + '.xml')
    outWidth = int(old_div(inWidth,inps.rlks))
    outLength = int(old_div(inLength,inps.alks))

    #look_msk infile outfile nrg nrlks nalks
    #run program here
    cmd = '{}/look_double {} {} {} {} {}'.format(BIN_PATH, inps.input, inps.output, inWidth, inps.rlks, inps.alks)
    runCmd(cmd)
    

    #get xml
    image = isceobj.createImage()
    accessMode = 'read'
    dataType = 'DOUBLE'
    width = outWidth
    image.initImage(inps.output,accessMode,width,dataType)

    image.addDescription('Pixel-by-pixel height in meters.')
    image.renderVRT()
    image.renderHdr()
    #image.finalizeImage()
    #image.createImage()



def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser( description='take looks')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help = 'input file')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help = 'output file')
    parser.add_argument('-r','--rlks', dest='rlks', type=int, default=4,
            help = 'number of range looks')
    parser.add_argument('-a','--alks', dest='alks', type=int, default=4,
            help = 'number of azimuth looks')
    return parser.parse_args()

if __name__ == '__main__':

    inps = cmdLineParse()

    if inps.input.endswith('.amp'):
        ampLooks(inps)
    elif inps.input.endswith('.int'):
        intLooks(inps)
    elif inps.input.endswith('.msk'):
        mskLooks(inps)
    elif inps.input.endswith('.hgt') or inps.input.endswith('.lat') or inps.input.endswith('.lon') or inps.input.find('lat') or inps.input.find('lon'):
        hgtLooks(inps)
    else:
        raise Exception('file type not supported yet')
        #sys.exit()

#look.py -i diff_20130927-20141211.int -o diff_20130927-20141211_16rlks_16alks.int -r 16 -a 16
#look.py -i 20130927-20141211.amp -o 20130927-20141211_16rlks_16alks.amp -r 16 -a 16    
#look.py -i msk.msk -o 3_6.msk -r 3 -a 6



