#!/bin/bash
cargo build --profile profiling
sudo ~/.cargo/bin/samply record -n target/profiling/filsimrs data/test2.dng
