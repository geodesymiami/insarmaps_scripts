#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alfredo Terrero, 2016                            #
############################################################


import os
import argparse
import glob
from add_attribute_insarmaps import (
    InsarDatabaseController
)
import json

# figure out what the name of the h5 file to put on site is in the given path
def get_H5_filename(path):
    return glob.glob("*.he5")[0]

def build_parser():
    dbHost = "insarmaps.rsmas.miami.edu"
    parser = argparse.ArgumentParser(description='Convert a Unavco format H5 file for ingestion into insarmaps.')
    required = parser.add_argument_group("required arguments")
    required.add_argument("-f", "--file", help="unavco file to ingest", required=False)
    required.add_argument("-u", "--user", help="username for the insarmaps database", required=True)
    required.add_argument("-p", "--password", help="password for the insarmaps database", required=True)
    required.add_argument("--host", default=dbHost, help="postgres DB URL for insarmaps database", required=True)
    required.add_argument("-U", "--server_user", help="username for the insarmaps server (the machine where the tileserver and http server reside)", required=True)
    parser.add_argument("-P", "--server_password", help="password for the insarmaps server (the machine where the tileserver and http server reside)", required=True)
    parser.add_argument("-d", "--disk_space", help="Get available disk space for database directory and tileserver directory", required=False, action="store_true")

    return parser

def main():
    parser = build_parser()
    parseArgs = parser.parse_args()

    dbUsername = parseArgs.user
    dbPassword = parseArgs.password
    dbHost = parseArgs.host
    serverUser = parseArgs.server_user
    serverPassword = parseArgs.server_password

    dbController = InsarDatabaseController(dbUsername,
                                           dbPassword,
                                           dbHost,
                                           'pgis')

    if parseArgs.disk_space:
        dbController.connect()
        json_data = dbController.get_disk_space()
        BYTES_IN_GIG = 1024 * 1024 * 1024
        for k, v in json_data.items():
            json_data[k] = str(round(v / BYTES_IN_GIG, 2)) + " G"
        print(json.dumps(json_data, indent=4, sort_keys=True))
        dbController.close()

    if parseArgs.file:
        bjobScriptFilename = "run_mintpy2insarmaps.py"

        path = parseArgs.file

        h5FileFullName = parseArgs.file

        curProjName = h5FileFullName.split(".")[0]

        jsonFolder = "json/"
        mbtilesFile = jsonFolder + curProjName + ".mbtiles"

# create working directory in scratch and copy relevant files over
        scratch_dir = os.environ["SCRATCHDIR"] + "/" + curProjName
        print("making directory " + scratch_dir)
        os.makedirs(scratch_dir)
        command = "cp " + h5FileFullName + " " + scratch_dir + "/"
        print("copying files to scratch with command " + command)
        os.system(command)

# go to scratch dir, and run the bjob command
        unavco2jsonCmd = "unavco2json_mbtiles.py " + h5FileFullName + " " + jsonFolder

        json2insarmapsCmd = "json_mbtiles2insarmaps.py " + \
            " -u " + dbUsername + " -p " + dbPassword + \
            " -U " + serverUser + " -P " + serverPassword + \
            " --mbtiles_file " + mbtilesFile + \
            " --json_folder " + jsonFolder + \
            " --host " + dbHost

        command = "echo '" + unavco2jsonCmd + " && " + json2insarmapsCmd + "' > " + bjobScriptFilename

        os.chdir(scratch_dir)
        os.system(command)

        os.system("createBatch.pl " + bjobScriptFilename)
        print("bjob finished")

if __name__ == '__main__':
    main()
