use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelExecuterError {
    #[error("Initialization failed: {0}")]
    InitializationError(String),
    #[error("Model loading failed: {0}")]
    LoadError(String),
    #[error("Inference execution failed: {0}")]
    ExecutionError(String),
    #[error("Invalid input tensor: {0}")]
    InputError(String),
    #[error("Invalid output tensor: {0}")]
    OutputError(String),
    #[error("Memory allocation failed: {0}")]
    AllocationError(String),
    #[error("FFI operation failed: {0}")]
    FFIError(String),
    #[error("Async channel error: {0}")]
    AsyncChannelError(String),
}
