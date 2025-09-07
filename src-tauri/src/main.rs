// src-tauri/src/main.rs

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{AppHandle, Emitter}; // FIX: Import the Emitter trait. Removed unused Manager.
use std::thread;
use serde_json::json;

// --- The Telemetry Listener ---
fn start_telemetry_listener(app: AppHandle) {
    thread::spawn(move || {
        let context = zmq::Context::new();
        let subscriber = context.socket(zmq::SUB).expect("Failed to create ZMQ subscriber");
        
        subscriber.connect("tcp://localhost:5556").expect("Failed to connect to telemetry publisher");
        subscriber.set_subscribe(b"").expect("Failed to subscribe to topics");
        
        println!("Telemetry listener started and subscribed.");

        loop {
            let _topic = subscriber.recv_string(0).unwrap().unwrap();
            let message_json_str = subscriber.recv_string(0).unwrap().unwrap();
            
            // The call to .emit() now works because the Emitter trait is in scope.
            app.emit("telemetry-update", &message_json_str).expect("Failed to emit event");
            
            println!("Received and forwarded a telemetry update.");
        }
    });
}

// --- The Tauri Command ---
#[tauri::command]
fn invoke_backend(query: String, gear_override: Option<String>) -> Result<String, String> {
    let context = zmq::Context::new();
    let requester = context.socket(zmq::REQ).map_err(|e| e.to_string())?;
    
    requester.connect("tcp://localhost:5555").map_err(|e| e.to_string())?;

    println!("Sending command to Python: Query='{}', Gear='{:?}'", &query, &gear_override);

    let command_payload = json!({
        "command": "reason",
        "payload": {
            "query": query,
            "gear_override": gear_override
        }
    });

    requester.send(&command_payload.to_string(), 0).map_err(|e| e.to_string())?;

    // --- FIX: Use recv_bytes for robustness, just like we learned before ---
    let response_bytes = requester.recv_bytes(0).map_err(|e| e.to_string())?;
    let response_str = String::from_utf8(response_bytes).map_err(|e| e.to_string())?;
    // --- END FIX ---
    
    println!("Received acknowledgment from Python: {}", &response_str);
    
    Ok(response_str)
}

// --- The Main Application Setup ---
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![invoke_backend])
        .setup(|app| {
            let handle = app.handle().clone();
            start_telemetry_listener(handle);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}