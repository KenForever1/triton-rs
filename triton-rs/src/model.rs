use crate::{check_err, Error};
use libc::{c_char, size_t};
use std::ffi::CStr;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::ptr;

pub struct Model {
    ptr: *mut triton_sys::TRITONBACKEND_Model,
}

impl Model {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, Error> {
        let mut model_name: *const c_char = ptr::null_mut();
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelName(self.ptr, &mut model_name) })?;

        let c_str = unsafe { CStr::from_ptr(model_name) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn version(&self) -> Result<u64, Error> {
        let mut version = 0u64;
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelVersion(self.ptr, &mut version) })?;
        Ok(version)
    }

    pub fn location(&self) -> Result<String, Error> {
        let mut artifact_type: triton_sys::TRITONBACKEND_ArtifactType = 0u32;
        let mut location: *const c_char = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelRepository(self.ptr, &mut artifact_type, &mut location)
        })?;

        let c_str = unsafe { CStr::from_ptr(location) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn path(&self, filename: &str) -> Result<PathBuf, Error> {
        Ok(PathBuf::from(format!(
            "{}/{}/{}",
            self.location()?,
            self.version()?,
            filename
        )))
    }

    pub fn load_file(&self, filename: &str) -> Result<Vec<u8>, Error> {
        let path = self.path(filename)?;
        let mut f = File::open(path)?;

        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;

        Ok(buffer)
    }

    pub fn model_config(&self) -> Result<String, Error> {
        // first step: call TRITONBACKEND_ModelConfig c func
        let config_version = 1;
        let mut msg : *mut triton_sys::TRITONSERVER_Message = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelConfig(self.ptr, config_version, &mut msg)
        })?;
    
        if msg.is_null() {
            return Err("Failed to get the message pointer".into());
        }
    
        // second step: call TRITONSERVER_MessageSerializeToJson c func, get json string from base and byte_size
        let mut base: *const libc::c_char = std::ptr::null();
        let mut byte_size: size_t = 0;
        check_err(unsafe {
            triton_sys::TRITONSERVER_MessageSerializeToJson(msg, &mut base, &mut byte_size)
        })?;
    
        if base.is_null() || byte_size == 0 {
            return Err("Failed to serialize the message to JSON".into());
        }
    
        // Convert C char array to Rust String
        let json_str = unsafe {
            std::ffi::CStr::from_ptr(base).to_string_lossy().into_owned()
        };
    
        Ok(json_str)
    }
}
