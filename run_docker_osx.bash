#!/bin/bash
# To learn more details for running GUI Docker apps on osx
# please reference https://goo.gl/SkN3pi
IP=$(ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p')
echo "adding IP $IP to xhost"
/opt/X11/bin/xhost + $IP

docker run --rm -it \
  --name capstone \
  -e DISPLAY=$IP:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $PWD:/capstone/ \
  -v /tmp/log:/root/.ros/ \
  capstone
