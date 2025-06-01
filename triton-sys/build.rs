use std::env;
use std::path::PathBuf;

fn main() {
    // println!("cargo:rustc-link-search=/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/lib");
    // println!("cargo:include=/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include/");
    let bindings = bindgen::Builder::default()
        .clang_arg("-Ideps/core/include")
        .clang_arg("-I/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include/")
        .clang_arg("-xc++")
        .clang_arg("--target=aarch64-unknown-linux-gnu")
        .header("deps/core/include/triton/core/tritonbackend.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
