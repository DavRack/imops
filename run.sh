#!/bin/bash
last_dng=$(ls ~/storage/dcim/Camera/ | grep dng | sort -r | sed 1q | sed 's/\.dng//')
cargo run --release /data/data/com.termux/files/home/storage/dcim/Camera/$last_dng.dng && cp result.jpg ~/storage/pictures/process_dng/$last_dng.jpg && termux-media-scan ~/storage/pictures/process_dng/
