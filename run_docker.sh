#!/usr/bin/env bash

sudo docker run -e server_ip=${server_ip} -p 80:80 -p 443:443 -p 5432:5432 -p 8888:8888 -it insarmaps_scripts:latest


