#! /usr/bin/env bash

kill -9 $(ps -ef|grep "gpt_model"|grep -v grep|awk '{print $2}')
