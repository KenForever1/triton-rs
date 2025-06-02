export PATH=$PATH:/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin
# export RUSTFLAGS="-L /path/to/target/libs"
cargo build --target=aarch64-unknown-linux-gnu --release
