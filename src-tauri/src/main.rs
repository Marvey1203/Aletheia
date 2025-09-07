// src-tauri/src/main.rs

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// This function will be callable from the frontend
#[tauri::command]
fn invoke_backend(query: String) -> Result<String, String> {
    let context = zmq::Context::new();
    let requester = context.socket(zmq::REQ).map_err(|e| e.to_string())?;

    requester
        .connect("tcp://localhost:5555")
        .map_err(|e| e.to_string())?;

    println!("Sending query to Python: {}", &query);

    requester
        .send(&query, 0)
        .map_err(|e| e.to_string())?;

    // --- FIX STARTS HERE ---
    // 1. Receive the response as raw bytes first. This is more robust.
    let response_bytes = requester.recv_bytes(0).map_err(|e| e.to_string())?;
    
    // 2. Try to convert the bytes into a valid UTF-8 String.
    //    This handles the error correctly if the backend sends invalid text.
    let final_response = String::from_utf8(response_bytes)
        .map_err(|e| e.to_string())?;
    // --- FIX ENDS HERE ---

    println!("Received response from Python.");
    Ok(final_response)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![invoke_backend])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}