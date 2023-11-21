#! /usr/bin/env bash

kill -9 $(ps -ef|grep "seq_parallel/bin/python3 -c from multiprocessing"|grep -v grep|awk '{print $2}')
