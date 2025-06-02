use crate::check_err;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

pub struct InferenceResponse {
    ptr: *mut triton_sys::TRITONSERVER_InferenceResponse,
}

impl InferenceResponse {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONSERVER_InferenceResponse) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_InferenceResponse {
        self.ptr
    }

    pub fn get_output_count(&self) -> Result<u32, Box<dyn std::error::Error>> {
        let mut count = 0;
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceResponseOutputCount(self.ptr, &mut count)
        })?;
        Ok(count)
    }

    pub fn get_output_data(&self, out_idx: u32) -> Result<OutputData, Box<dyn std::error::Error>> {
        OutputData::get_output_data(self.ptr, out_idx)
    }
}

pub struct OutputData {
    pub name: String,
    pub datatype: triton_sys::TRITONSERVER_DataType,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
    pub memory_type: triton_sys::TRITONSERVER_MemoryType,
    pub memory_type_id: i64,
}

impl OutputData {
    pub fn get_output_data(
        response: *mut triton_sys::TRITONSERVER_InferenceResponse,
        out_idx: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let mut name_ptr: *const c_char = std::ptr::null();
            let mut datatype = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32;
            let mut shape_ptr: *const i64 = std::ptr::null();
            let mut dim_count: u64 = 0;
            let mut base_ptr: *const c_void = std::ptr::null();
            let mut byte_size: usize = 0;
            let mut memory_type = triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
            let mut memory_type_id: i64 = 0;

            let err = triton_sys::TRITONSERVER_InferenceResponseOutput(
                response,
                out_idx,
                &mut name_ptr,
                &mut datatype,
                &mut shape_ptr,
                &mut dim_count,
                &mut base_ptr,
                &mut byte_size,
                &mut memory_type,
                &mut memory_type_id,
                std::ptr::null_mut(),
            );

            if !err.is_null() {
                return Err("Failed to get output".into());
            }

            let name = CStr::from_ptr(name_ptr).to_str()?.to_owned();
            let shape = if !shape_ptr.is_null() {
                std::slice::from_raw_parts(shape_ptr, dim_count as usize).to_vec()
            } else {
                vec![]
            };

            let data = if !base_ptr.is_null() {
                std::slice::from_raw_parts(base_ptr as *const u8, byte_size).to_vec()
            } else {
                vec![]
            };

            Ok(Self {
                name,
                datatype,
                shape,
                data,
                memory_type,
                memory_type_id,
            })
        }
    }

    pub fn print_info(&self) {
        println!(
            "name: {:?}, datatype: {:?}, shape: {:?}, data len: {:?}, memory_type: {:?}, memory_type_id: {:?}",
            self.name, self.datatype, self.shape, self.data.len(), self.memory_type, self.memory_type_id);
        if self.datatype == triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES {
            println!("data: {:?}", String::from_utf8_lossy(&self.data));
        }
    }
}
