// FILE: src-tauri/src/main.rs

#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]

use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use tauri::{Manager, State};

// A struct to hold the ZMQ socket so we can share it across commands.
struct ZmqConnection {
  socket: Mutex<zmq::Socket>,
}

// A struct to hold the handle to our Python sidecar process.
struct SidecarProcess {
    handle: Arc<Mutex<Child>>,
}

fn main() {
    // --- 1. Set up the ZMQ Requester Socket ---
    let context = zmq::Context::new();
    let requester = context.socket(zmq::REQ).expect("Failed to create ZMQ socket");
    requester.connect("tcp://127.0.0.1:5555").expect("Failed to connect to ZMQ socket");
    
    tauri::Builder::default()
        // --- 2. Add the ZMQ connection to Tauri's state management ---
        .manage(ZmqConnection { socket: Mutex::new(requester) })
        .setup(|app| {
            println!("Spawning Python sidecar...");
            
            // --- 3. Spawn the Python IPC server as a sidecar ---
            // This command assumes you run `cargo tauri dev` from the `src-tauri` directory.
            // The `../` correctly points back to the project root.
            let child = Command::new("python3")
                .arg("../interfaces/ipc_server.py")
                .arg("--dummy") // Start in dummy mode for easier development
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("Failed to spawn Python sidecar. Is python in your PATH?");
            
            let handle = Arc::new(Mutex::new(child));
            app.manage(SidecarProcess { handle: handle.clone() });
            
            Ok(())
        })
        // --- 4. Define the `reason` command that our JS frontend will call ---
        .invoke_handler(tauri::generate_handler![reason])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
fn reason(query: String, state: State<ZmqConnection>) -> Result<serde_json::Value, String> {
    println!("Frontend requested reason for query: \"{}\"", query);
    
    let socket = state.socket.lock().map_err(|e| format!("ZMQ socket lock failed: {:?}", e))?;

    socket.send(&query, 0).map_err(|e| format!("ZMQ send failed: {:?}", e))?;

    // 1. Receive raw bytes from the socket. This is more robust.
    let response_bytes = socket.recv_bytes(0).map_err(|e| format!("ZMQ receive failed: {:?}", e))?;

    // 2. Convert the bytes to a UTF-8 string, which can fail.
    let response_string = String::from_utf8(response_bytes)
        .map_err(|e| format!("Failed to decode UTF-8 response from Python: {:?}", e))?;

    // 3. Parse the valid string into a JSON value.
    serde_json::from_str(&response_string).map_err(|e| format!("Failed to parse JSON response: {:?}", e))
}