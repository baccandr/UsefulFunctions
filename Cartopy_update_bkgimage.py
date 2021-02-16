# -*- coding: utf-8 -*-
"""
Update cartopy background image.
Script adapted from: https://thomasguymer.co.uk/blog/2018/2018-01-15/

it requires: https://imagemagick.org/script/index.php
and: http://optipng.sourceforge.net/

in order to find the default folder for cartopy bkg images you can use

import cartopy
import os

# Print path ...
print os.path.join(cartopy.__path__[0], "data", "raster", "natural_earth")

@author: baccarini_a
"""

#%% Import modules
import json
import os
import requests
import subprocess
import zipfile
import email.utils

#%% Define the background image datasets
imgs = {
    "cross-blend-hypso" : {
        "description" : "Cross-blended Hypsometric Tints with Relief, Water, Drains and Ocean Bottom from Natural Earth",
        "rasters" : {
            "large"  : "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/HYP_HR_SR_OB_DR.zip",
            "medium" : "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/HYP_LR_SR_OB_DR.zip",
            "small"  : "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/raster/HYP_50M_SR_W.zip",
        },
        "source" : "https://www.naturalearthdata.com/downloads/10m-raster-data/10m-cross-blend-hypso/ and https://www.naturalearthdata.com/downloads/50m-raster-data/50m-cross-blend-hypso/",
    },
    "natural-earth-2" : {
        "description" : "Natural Earth II with Shaded Relief, Water and Drainages from Natural Earth",
        "rasters" : {
            "large"  : "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/NE2_HR_LC_SR_W_DR.zip",
            "medium" : "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/NE2_LR_LC_SR_W_DR.zip",
            "small"  : "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/raster/NE2_50M_SR_W.zip",
        },
        "source" : "https://www.naturalearthdata.com/downloads/10m-raster-data/10m-natural-earth-2/ and https://www.naturalearthdata.com/downloads/50m-raster-data/50m-natural-earth-2/",
    },
}

#%% Def functions (from https://github.com/Guymer/)

def download(sess, method, url):
    # Try to download the URL and catch common errors ...
    try:
        resp = sess.request(method, url, timeout = 10.0)
    except requests.exceptions.TooManyRedirects:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False

    # Exit if the response was bad ...
    if resp.status_code != 200:
        return False

    return resp
def download_file(sess, url, fname):

    # Try to download the file and catch common errors ...
    resp = download(sess, "get", url)
    if resp is False:
        return False

    # Save file to disk ...
    dname = os.path.dirname(fname)
    if len(dname) > 0:
        if not os.path.exists(dname):
            os.makedirs(dname)
    open(fname, "wb").write(resp.content)

    # Change modification time if present ...
    if "Last-Modified" in resp.headers:
        modtime = email.utils.mktime_tz(email.utils.parsedate_tz(resp.headers["Last-Modified"]))
        os.utime(fname, (modtime, modtime))

    return True

#%% Main script

# Create JSON dictionary ...
data = {}
data["__comment__"] = "JSON file specifying the image to use for a given type/name and resolution. Read in by cartopy.mpl.geoaxes.read_user_background_images."

# ******************************************************************************
# *                   CREATE PNG IMAGES FROM REMOTE SOURCES                    *
# ******************************************************************************

# Start session ...
sess = requests.Session()
sess.allow_redirects = True
sess.headers.update({"Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
sess.headers.update({"Accept-Language" : "en-GB,en;q=0.5"})
sess.headers.update({"Accept-Encoding" : "gzip, deflate"})
sess.headers.update({"DNT" : "1"})
sess.headers.update({"Upgrade-Insecure-Requests" : "1"})
sess.headers.update({"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:68.0) Gecko/20100101 Firefox/68.0"})
sess.max_redirects = 5

# Loop over background image datasets ...
for img in imgs.keys():
    # Add to JSON dictionary ...
    data[img] = {}
    data[img]["__comment__"] = imgs[img]["description"]
    data[img]["__projection__"] = "PlateCarree"
    data[img]["__source__"] = imgs[img]["source"]

    # Loop over sizes ...
    for size in imgs[img]["rasters"].keys():
        # Deduce ZIP file name and download it if it is missing ...
        zfile = "{0:s}_{1:s}.zip".format(img, size)
        if not os.path.exists(zfile):
            print("Downloading \"{0:s}\" ...".format(zfile))
            if not download_file(sess, imgs[img]["rasters"][size], zfile):
                raise Exception("download failed", imgs[img]["rasters"][size]) from None

        # Deduce TIF file name and extract it if is missing ...
        tfile = "{0:s}_{1:s}.tif".format(img, size)
        if not os.path.exists(tfile):
            print("Extracting \"{0:s}\" ...".format(tfile))
            with zipfile.ZipFile(zfile, "r") as zobj:
                for member in zobj.namelist():
                    if member.lower().endswith(".tif"):
                        tmp = zobj.extract(member)
                        os.rename(tmp, tfile)
                        break

        # Deduce PNG file name and convert TIF to PNG if it is missing ...
        pfile = "{0:s}_{1:s}.png".format(img, size)
        if not os.path.exists(pfile):
            print("Creating \"{0:s}\" ...".format(pfile))
            subprocess.run(
                ["C:\\Program Files\\ImageMagick-7.0.11-Q16-HDRI\\convert.exe",
                 tfile, pfile],
                stderr = open(os.devnull, "wt"),
                stdout = open(os.devnull, "wt")
            )
            subprocess.run(
                ["optipng", pfile],
                stderr = open(os.devnull, "wt"),
                stdout = open(os.devnull, "wt")
            )

        # Add to JSON dictionary ...
        data[img][size] = pfile

# End session ...
sess.close()

# ******************************************************************************
# *              CREATE DOWNSCALED PNG IMAGES FROM LOCAL SOURCES               *
# ******************************************************************************

# Loop over background image datasets ...
for img in imgs.keys():
    # Loop over sizes ...
    for size in imgs[img]["rasters"].keys():
        # Deduce PNG file name ...
        pfile1 = "{0:s}_{1:s}.png".format(img, size)

        # Loop over downscaled sizes ...
        for width in [512, 1024, 2048, 4096]:
            # Deduce downscaled PNG file name and create it if missing ...
            pfile2 = "{0:s}_{1:s}{2:04d}px.png".format(img, size, width)
            if not os.path.exists(pfile2):
                print("Creating \"{0:s}\" ...".format(pfile2))
                subprocess.run(
                    ["C:\\Program Files\\ImageMagick-7.0.11-Q16-HDRI\\convert.exe",
                     pfile1, "-resize", "{0:d}x".format(width), pfile2],
                    stderr = open(os.devnull, "wt"),
                    stdout = open(os.devnull, "wt")
                )
                subprocess.run(
                    ["optipng", pfile2],
                    stderr = open(os.devnull, "wt"),
                    stdout = open(os.devnull, "wt")
                )

            # Add to JSON dictionary ...
            data[img]["{0:s}{1:04d}px".format(size, width)] = pfile2

# Save JSON dictionary ...
open(
    "images.json",
    "wt",
).write(
    json.dumps(
        data,
        indent = 4,
        sort_keys = True
    )
)

