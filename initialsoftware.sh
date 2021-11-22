#!/bin/bash

sudo apt-get update
sudo apt-get -y install build-essential libssl-dev python-dev libffi-dev python-pip git
sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

sudo pip3 install -r requirements.txt

version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
