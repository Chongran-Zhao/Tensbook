//! TensorForge desktop app: a thin Tauri shell around the `tensorforge`
//! engine crate. The frontend (../ui) sends `.tens` source via the
//! `run_tens` command and receives structured outputs for KaTeX rendering.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use tauri_plugin_dialog::DialogExt;

#[derive(Serialize)]
struct RunOutput {
    header: String,
    latex: String,
    line: usize,
    error: Option<String>,
}

#[derive(Serialize)]
struct RunResult {
    ok: bool,
    outputs: Vec<RunOutput>,
    error: Option<String>,
}

/// Result of an open dialog: `None` if the user cancelled.
#[derive(Serialize)]
struct OpenedFile {
    path: String,
    source: String,
}

/// Parse and execute a `.tens` program with per-statement error recovery:
/// each failing statement yields an error output (tagged with its source
/// line) and execution continues. A parse error still fails the whole run
/// (the rest of the file can't be tokenized reliably).
#[tauri::command]
fn run_tens(source: String) -> RunResult {
    let stmts = match tensorforge::parser::parse(&source) {
        Ok(stmts) => stmts,
        Err(e) => {
            return RunResult {
                ok: false,
                outputs: vec![],
                error: Some(e.to_string()),
            }
        }
    };
    let outputs = tensorforge::interpreter::Interpreter::new().run_lenient(&stmts);
    RunResult {
        ok: true,
        outputs: outputs
            .into_iter()
            .map(|o| RunOutput {
                header: o.header,
                latex: o.latex,
                line: o.line,
                error: o.error,
            })
            .collect(),
        error: None,
    }
}

/// Show a native open dialog and read the chosen `.tens` file.
/// Returns `Ok(None)` if the user cancelled.
///
/// `async` is required: sync commands run on the main thread, where
/// `blocking_pick_file` would deadlock waiting for the dialog. Async
/// commands run on a worker thread, making the blocking call safe.
#[tauri::command]
async fn open_tens(app: tauri::AppHandle) -> Result<Option<OpenedFile>, String> {
    let Some(path) = app
        .dialog()
        .file()
        .add_filter("TensorForge", &["tens"])
        .blocking_pick_file()
    else {
        return Ok(None);
    };
    let path = path.into_path().map_err(|e| e.to_string())?;
    let source = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    Ok(Some(OpenedFile {
        path: path.display().to_string(),
        source,
    }))
}

/// Save the source. `path: None` (or Save As) opens a native save dialog.
/// Returns the path written, or `None` if the user cancelled.
/// `async` for the same main-thread-deadlock reason as [`open_tens`].
#[tauri::command]
async fn save_tens(
    app: tauri::AppHandle,
    source: String,
    path: Option<String>,
) -> Result<Option<String>, String> {
    let target = match path {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let Some(picked) = app
                .dialog()
                .file()
                .add_filter("TensorForge", &["tens"])
                .set_file_name("untitled.tens")
                .blocking_save_file()
            else {
                return Ok(None);
            };
            picked.into_path().map_err(|e| e.to_string())?
        }
    };
    std::fs::write(&target, source).map_err(|e| e.to_string())?;
    Ok(Some(target.display().to_string()))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![run_tens, open_tens, save_tens])
        .run(tauri::generate_context!())
        .expect("error while running TensorForge");
}
