//! TensorForge desktop app: a thin Tauri shell around the `tensorforge`
//! engine crate. The frontend (../ui) sends `.tens` source via the
//! `run_tens` command and receives structured outputs for KaTeX rendering.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;

#[derive(Serialize)]
struct RunOutput {
    header: String,
    latex: String,
}

#[derive(Serialize)]
struct RunResult {
    ok: bool,
    outputs: Vec<RunOutput>,
    error: Option<String>,
}

/// Parse and execute a `.tens` program; never fails the IPC call itself —
/// engine errors come back in `error` so the UI can show them inline.
#[tauri::command]
fn run_tens(source: String) -> RunResult {
    match tensorforge::run_source(&source) {
        Ok(outputs) => RunResult {
            ok: true,
            outputs: outputs
                .into_iter()
                .map(|o| RunOutput {
                    header: o.header,
                    latex: o.latex,
                })
                .collect(),
            error: None,
        },
        Err(e) => RunResult {
            ok: false,
            outputs: vec![],
            error: Some(e.to_string()),
        },
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![run_tens])
        .run(tauri::generate_context!())
        .expect("error while running TensorForge");
}
