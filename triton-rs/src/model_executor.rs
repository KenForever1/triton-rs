//! A Rust-style Triton Server model executor with FFI bindings

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;
// use async_trait::async_trait;
use crate::error::ModelExecuterError;
use crate::{InferenceRequest, dump_err};
use tokio::sync::oneshot;

// // #[async_trait]
// pub trait ModelExecuter {
//     fn load_model(&mut self, model_path: &str) -> Result<(), ModelExecuterError>;
//     async fn execute(
//         &self,
//         request: *mut triton_sys::TRITONSERVER_InferenceRequest,
//     ) -> Result<*mut triton_sys::TRITONSERVER_InferenceResponse, ModelExecuterError>;
// }

pub struct TritonModelExecuter {
    server: *mut triton_sys::TRITONSERVER_Server,
    allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
}

impl Drop for TritonModelExecuter {
    fn drop(&mut self) {
        unsafe {
            // Clean up allocator when executor is dropped
            if !self.allocator.is_null() {
                Box::from_raw(self.allocator);
            }
        }
    }
}

impl TritonModelExecuter {
    pub fn new(server: *mut triton_sys::TRITONSERVER_Server) -> Result<Self, ModelExecuterError> {
        let mut allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator = ptr::null_mut();

        unsafe {
            let err = triton_sys::TRITONSERVER_ResponseAllocatorNew(
                &mut allocator,
                Some(response_alloc),
                Some(response_release),
                None,
            );

            if !err.is_null() {
                dump_err(err);
                return Err(ModelExecuterError::InitializationError(
                    "Failed to create response allocator".to_string(),
                ));
            }
        }

        Ok(Self { server, allocator })
    }
}

// #[async_trait]
impl TritonModelExecuter {
    pub async fn execute(
        &self,
        request: &InferenceRequest,
    ) -> Result<*mut triton_sys::TRITONSERVER_InferenceResponse, ModelExecuterError> {
        let (tx, rx) = oneshot::channel();
        let tx_ptr = Box::into_raw(Box::new(tx));

        unsafe {
            // Set response callback
            let err = triton_sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                request.as_ptr(),
                self.allocator,
                ptr::null_mut(),
                Some(infer_response_complete),
                tx_ptr as *mut c_void,
            );

            if !err.is_null() {
                dump_err(err);
                return Err(ModelExecuterError::ExecutionError(
                    "Failed to set response callback".to_string(),
                ));
            }

            // Start async inference
            let err =
                triton_sys::TRITONSERVER_ServerInferAsync(self.server, request.as_ptr(), ptr::null_mut());

            if !err.is_null() {
                dump_err(err);
                return Err(ModelExecuterError::ExecutionError(
                    "Failed to start async inference".to_string(),
                ));
            }
        }

        // Wait for response
        rx.await.map_err(|_| {
            ModelExecuterError::ExecutionError("Failed to receive inference response".to_string())
        })
    }
}

// FFI callback functions
extern "C" fn infer_response_complete(
    response: *mut triton_sys::TRITONSERVER_InferenceResponse,
    flags: u32,
    userp: *mut c_void,
) {
    if !response.is_null() {
        let tx = unsafe {
            Box::from_raw(
                userp as *mut oneshot::Sender<*mut triton_sys::TRITONSERVER_InferenceResponse>,
            )
        };
        let _ = tx.send(response);
    }
}

extern "C" fn response_alloc(
    allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    tensor_name: *const libc::c_char,
    byte_size: libc::size_t,
    preferred_memory_type: triton_sys::TRITONSERVER_MemoryType,
    preferred_memory_type_id: i64,
    userp: *mut c_void,
    buffer: *mut *mut c_void,
    buffer_userp: *mut *mut c_void,
    actual_memory_type: *mut triton_sys::TRITONSERVER_MemoryType,
    actual_memory_type_id: *mut i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    unsafe {
        *actual_memory_type = preferred_memory_type;
        *actual_memory_type_id = preferred_memory_type_id;

        if byte_size == 0 {
            *buffer = ptr::null_mut();
            *buffer_userp = ptr::null_mut();
            return ptr::null_mut();
        }

        let allocated_ptr = match preferred_memory_type {
            // Implement memory allocation based on type (CPU/GPU)
            _ => libc::malloc(byte_size),
        };

        if !allocated_ptr.is_null() {
            *buffer = allocated_ptr;
            *buffer_userp = Box::into_raw(Box::new(CString::from_vec_unchecked(
                unsafe { std::ffi::CStr::from_ptr(tensor_name) }
                    .to_bytes()
                    .to_vec(),
            ))) as *mut c_void;
        }

        ptr::null_mut() // Success
    }
}

extern "C" fn response_release(
    allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    buffer: *mut c_void,
    buffer_userp: *mut c_void,
    byte_size: libc::size_t,
    memory_type: triton_sys::TRITONSERVER_MemoryType,
    memory_type_id: i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    unsafe {
        if !buffer.is_null() {
            match memory_type {
                triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU => {
                    libc::free(buffer)
                }
                _ => libc::free(buffer), // Simplified for example
            };
        }

        if !buffer_userp.is_null() {
            drop(Box::from_raw(buffer_userp as *mut CString));
        }

        ptr::null_mut() // Success
    }
}

pub fn new_executer(
    server: *mut triton_sys::TRITONSERVER_Server,
) -> Result<Arc<TritonModelExecuter>, ModelExecuterError> {
    let executor = TritonModelExecuter::new(server)?;
    Ok(Arc::new(executor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use tokio::test;

    // #[test]
    // async fn test_executer_creation() {
    //     // Create with null server pointer for test
    //     let executer = new_executer(ptr::null_mut());
    //     assert!(executer.is_ok());

    //     let executer = executer.unwrap();
    // }
}
