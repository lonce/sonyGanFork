#!/bin/bash
# get data lines ; remove progress info, remove duplicate lines
cat $1 |  grep "Iter:" |  cut -d ' ' -f 1-28 | awk '{if ($0!=prev) print ; prev=$0}'