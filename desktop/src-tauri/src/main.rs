// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::prelude::*;
use std::os::unix::net::UnixStream;
use std::{sync::Mutex};
use tauri::State;

#[tauri::command]
fn write_message(message: &str, c: State<IPCClient>) {
    let mut client = c.client.lock().unwrap();
    client.write_all(message.as_bytes()).unwrap();
}

#[tauri::command]
fn read_message(c: State<IPCClient>) -> String {
    let mut client = c.client.lock().unwrap();
    let mut buff = [0; 1024];
    let count = client.read(&mut buff).unwrap();
    let response = String::from_utf8(buff[..count].to_vec()).unwrap();
    return response
}


#[tauri::command]
fn write_and_wait_for_response_blocking(message: &str, c: State<IPCClient>) -> String {
    let mut client = c.client.lock().unwrap();
    client.write_all(message.as_bytes()).unwrap();
    let mut buff = [0; 1024];
    let count = client.read(&mut buff).unwrap();
    let response = String::from_utf8(buff[..count].to_vec()).unwrap();
    return response
}

struct IPCClient {
    client: Mutex<UnixStream>
}

fn main() {
    tauri::Builder::default()
        .manage(IPCClient { client: Mutex::new(UnixStream::connect("/tmp/jr.sock").unwrap())})
        .invoke_handler(tauri::generate_handler![write_message, read_message, write_and_wait_for_response_blocking])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
