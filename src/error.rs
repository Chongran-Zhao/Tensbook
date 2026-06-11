use std::fmt;

/// Unified error type for parsing, type checking and evaluation.
#[derive(Debug, Clone)]
pub struct Error {
    pub message: String,
    pub line: Option<usize>,
}

impl Error {
    pub fn new(message: impl Into<String>, line: Option<usize>) -> Self {
        Error {
            message: message.into(),
            line,
        }
    }

    pub fn msg(message: impl Into<String>) -> Self {
        Error::new(message, None)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "error (line {}): {}", line, self.message),
            None => write!(f, "error: {}", self.message),
        }
    }
}

impl std::error::Error for Error {}
