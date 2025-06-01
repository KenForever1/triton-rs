//! https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api

use libc::c_void;
use std::ffi::CString;
use std::ptr;
use std::slice;
use triton_rs::Backend;

pub(crate) type BoxError = Box<dyn std::error::Error>;

pub(crate) fn check_err(err: *mut triton_sys::TRITONSERVER_Error) -> Result<(), BoxError> {
    if !err.is_null() {
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

struct ExampleBackend;

impl Backend for ExampleBackend {
    fn model_instance_execute(
        model: triton_rs::Model,
        requests: &[triton_rs::Request],
    ) -> Result<(), triton_rs::Error> {
        println!("[EXAMPLE] model_instance_execute");

        println!("[EXAMPLE] model config: {:?}", model.model_config()?);

        println!(
            "[EXAMPLE] request for model {} {} {}",
            model.name()?,
            model.version()?,
            model.location()?
        );

        for request in requests {
            let mut response: *mut triton_sys::TRITONBACKEND_Response = ptr::null_mut();
            check_err(unsafe {
                triton_sys::TRITONBACKEND_ResponseNew(&mut response, request.as_ptr())
            })?;

            let prompt = request.get_input("prompt")?.as_string()?;
            println!("[EXAMPLE] prompt: {prompt}");

            let out = format!("you said: {prompt}");
            let encoded = triton_rs::encode_string(&out);

            {
                let mut output: *mut triton_sys::TRITONBACKEND_Output = ptr::null_mut();
                let name = CString::new("output").expect("CString::new failed");
                let datatype = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES;
                let shape = &[1];
                let dims_count = 1;

                check_err(unsafe {
                    triton_sys::TRITONBACKEND_ResponseOutput(
                        response,
                        &mut output,
                        name.as_ptr(),
                        datatype,
                        shape.as_ptr(),
                        dims_count,
                    )
                })?;

                {
                    let mut buffer: *mut c_void = ptr::null_mut();
                    let buffer_byte_size = encoded.len() as u64;
                    let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
                    let mut memory_type_id = 0;
                    check_err(unsafe {
                        triton_sys::TRITONBACKEND_OutputBuffer(
                            output,
                            &mut buffer,
                            buffer_byte_size,
                            &mut memory_type,
                            &mut memory_type_id,
                        )
                    })?;

                    let mem: &mut [u8] = unsafe {
                        slice::from_raw_parts_mut(buffer as *mut u8, buffer_byte_size as usize)
                    };

                    mem.copy_from_slice(&encoded);
                }
            }

            let send_flags =
            triton_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL;
            let err = ptr::null_mut();
            check_err(unsafe {
                triton_sys::TRITONBACKEND_ResponseSend(&mut *response, send_flags, err)
            })?;
        }

        Ok(())
    }
}

triton_rs::declare_backend!(ExampleBackend);
