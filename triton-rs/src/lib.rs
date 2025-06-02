mod backend;
mod model;
mod request;
mod model_executor;
mod error;
mod inference_request;
mod inference_response;

pub use backend::Backend;
pub use model::Model;
pub use request::Request;
pub use triton_sys as sys;
pub use model_executor::TritonModelExecuter;
pub use inference_request::InferenceRequest;
pub use inference_response::InferenceResponse;
pub use error::ModelExecuterError;

pub type Error = Box<dyn std::error::Error>;

pub(crate) fn check_err(err: *mut triton_sys::TRITONSERVER_Error) -> Result<(), Error> {
    if !err.is_null() {
        unsafe{
            let err_msg = triton_sys::TRITONSERVER_ErrorCodeString(err);
            eprintln!(
                "check err : {:?}",
                std::ffi::CStr::from_ptr(err_msg)
            );
            triton_sys::TRITONSERVER_ErrorDelete(err);
        }

        let code = unsafe { triton_sys::TRITONSERVER_ErrorCode(err) };
        Err(format!(
            "TRITONBACKEND_ModelInstanceModel returned error code {}",
            code
        )
        .into())
    } else {
        Ok(())
    }
}

pub(crate) fn dump_err(err: *mut triton_sys::TRITONSERVER_Error) {
    if !err.is_null() {
        unsafe{
            let err_msg = triton_sys::TRITONSERVER_ErrorCodeString(err);
            eprintln!(
                "check err : {:?}",
                std::ffi::CStr::from_ptr(err_msg)
            );
            triton_sys::TRITONSERVER_ErrorDelete(err);
        }
    }
}

pub fn decode_string(data: &[u8]) -> Result<Vec<String>, Error> {
    let mut strings = vec![];
    let mut i = 0;

    while i < data.len() {
        // l = struct.unpack_from("<I", val_buf, offset)[0]
        // offset += 4
        let wide = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;

        // sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        // offset += l
        // strs.append(sb)
        let bytes = &data[i..i + wide];
        let string = String::from_utf8_lossy(bytes).to_string();
        i += wide;

        strings.push(string);
    }

    Ok(strings)
}

pub fn encode_string(value: &str) -> Vec<u8> {
    let mut bytes = vec![];

    let value: Vec<u8> = value.bytes().collect();

    // l = struct.unpack_from("<I", val_buf, offset)[0]
    // offset += 4
    let len = (value.len() as u32).to_le_bytes();
    bytes.extend_from_slice(&len);

    // sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
    // offset += l
    // strs.append(sb)
    bytes.extend_from_slice(&value);

    bytes
}
