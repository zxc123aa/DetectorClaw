use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::sync::Mutex;

#[allow(dead_code)]
#[path = "main.rs"]
mod core;

static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

fn set_last_error(message: impl Into<String>) {
    *LAST_ERROR.lock().expect("last error mutex poisoned") = Some(message.into());
}

fn clear_last_error() {
    *LAST_ERROR.lock().expect("last error mutex poisoned") = None;
}

fn into_raw_string(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(string) => string.into_raw(),
        Err(_) => CString::new("native string contained interior NUL")
            .expect("fallback CString must be valid")
            .into_raw(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rcf_segment_detect_json(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        set_last_error("null input passed to rcf_segment_detect_json");
        return ptr::null_mut();
    }

    let input_text = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(value) => value,
        Err(error) => {
            set_last_error(format!("invalid utf-8 input: {error}"));
            return ptr::null_mut();
        }
    };

    match core::segment_detect_json(input_text) {
        Ok(output) => {
            clear_last_error();
            into_raw_string(output)
        }
        Err(error) => {
            set_last_error(error.to_string());
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rcf_preview_command_json(
    command: *const c_char,
    input: *const c_char,
) -> *mut c_char {
    if command.is_null() || input.is_null() {
        set_last_error("null argument passed to rcf_preview_command_json");
        return ptr::null_mut();
    }

    let command_text = match unsafe { CStr::from_ptr(command) }.to_str() {
        Ok(value) => value,
        Err(error) => {
            set_last_error(format!("invalid utf-8 command: {error}"));
            return ptr::null_mut();
        }
    };
    let input_text = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(value) => value,
        Err(error) => {
            set_last_error(format!("invalid utf-8 input: {error}"));
            return ptr::null_mut();
        }
    };

    match core::preview_command_json(command_text, input_text) {
        Ok(output) => {
            clear_last_error();
            into_raw_string(output)
        }
        Err(error) => {
            set_last_error(error.to_string());
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rcf_dose_batch_json(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        set_last_error("null input passed to rcf_dose_batch_json");
        return ptr::null_mut();
    }

    let input_text = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(value) => value,
        Err(error) => {
            set_last_error(format!("invalid utf-8 input: {error}"));
            return ptr::null_mut();
        }
    };

    match core::dose_batch_json(input_text) {
        Ok(output) => {
            clear_last_error();
            into_raw_string(output)
        }
        Err(error) => {
            set_last_error(error.to_string());
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rcf_last_error_message() -> *mut c_char {
    let message = LAST_ERROR
        .lock()
        .expect("last error mutex poisoned")
        .clone()
        .unwrap_or_default();
    into_raw_string(message)
}

#[unsafe(no_mangle)]
pub extern "C" fn rcf_free_string(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(value));
    }
}
