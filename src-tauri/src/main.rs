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
    row: Option<usize>,
}

#[derive(Serialize)]
struct RunResult {
    ok: bool,
    outputs: Vec<RunOutput>,
    error: Option<String>,
}

/// Result of an open dialog: `None` if the user cancelled.
/// `save_path` is `None` for imported Markdown so Save becomes Save As .tens.
/// `files` lists the sibling `.tens` files of the opened file's folder so
/// the frontend can show a file rail.
#[derive(Serialize)]
struct OpenedFile {
    path: String,
    save_path: Option<String>,
    source: String,
    folder: Option<String>,
    files: Vec<FileEntry>,
}

#[derive(Serialize)]
struct FileEntry {
    name: String,
    path: String,
}

/// All `.tens` files directly inside `dir`, sorted by name.
fn list_tens_files(dir: &std::path::Path) -> Vec<FileEntry> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<FileEntry> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let is_tens = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("tens"));
            if !path.is_file() || !is_tens {
                return None;
            }
            Some(FileEntry {
                name: path.file_name()?.to_string_lossy().into_owned(),
                path: path.display().to_string(),
            })
        })
        .collect();
    files.sort_by(|a, b| a.name.cmp(&b.name));
    files
}

fn is_tens_path(path: &std::path::Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tens"))
}

fn with_tens_extension(mut path: std::path::PathBuf) -> std::path::PathBuf {
    if !is_tens_path(&path) {
        path.set_extension("tens");
    }
    path
}

fn opened_file(path: std::path::PathBuf) -> Result<OpenedFile, String> {
    let source = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let folder = path.parent().map(|p| p.to_path_buf());
    let save_path = is_tens_path(&path).then(|| path.display().to_string());
    Ok(OpenedFile {
        path: path.display().to_string(),
        save_path,
        source,
        folder: folder.as_ref().map(|p| p.display().to_string()),
        files: folder.as_deref().map(list_tens_files).unwrap_or_default(),
    })
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
                row: o.row,
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
        .add_filter("TensorForge / Markdown", &["tens", "md", "markdown"])
        .add_filter("TensorForge", &["tens"])
        .add_filter("Markdown", &["md", "markdown"])
        .blocking_pick_file()
    else {
        return Ok(None);
    };
    let path = path.into_path().map_err(|e| e.to_string())?;
    opened_file(path).map(Some)
}

/// Read a specific `.tens` file (file-rail click), re-listing its folder.
#[tauri::command]
async fn read_tens(path: String) -> Result<OpenedFile, String> {
    let path = std::path::PathBuf::from(path);
    if !is_tens_path(&path) {
        return Err("read_tens only accepts .tens files".to_string());
    }
    opened_file(path)
}

/// Re-list a folder (file-rail restore on startup).
#[tauri::command]
async fn list_folder(path: String) -> Result<Vec<FileEntry>, String> {
    Ok(list_tens_files(std::path::Path::new(&path)))
}

/// Show the native print dialog for the current window content. JS
/// `window.print()` is not reliably wired in WKWebView; this is the
/// supported path.
#[tauri::command]
async fn print_window(window: tauri::WebviewWindow) -> Result<(), String> {
    window.print().map_err(|e| e.to_string())
}

/// Save the source. `path: None` (or Save As) opens a native save dialog.
/// Returns the path written, or `None` if the user cancelled.
/// `async` for the same main-thread-deadlock reason as [`open_tens`].
#[tauri::command]
async fn save_tens(
    app: tauri::AppHandle,
    source: String,
    path: Option<String>,
    default_filename: Option<String>,
) -> Result<Option<String>, String> {
    let target = match path {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let Some(picked) = app
                .dialog()
                .file()
                .add_filter("TensorForge", &["tens"])
                .set_file_name(default_filename.as_deref().unwrap_or("untitled.tens"))
                .blocking_save_file()
            else {
                return Ok(None);
            };
            picked.into_path().map_err(|e| e.to_string())?
        }
    };
    let target = with_tens_extension(target);
    std::fs::write(&target, source).map_err(|e| e.to_string())?;
    Ok(Some(target.display().to_string()))
}

/// Export a generated text document such as Markdown.
#[tauri::command]
async fn export_text(
    app: tauri::AppHandle,
    content: String,
    default_filename: String,
    filter_name: String,
    extensions: Vec<String>,
) -> Result<Option<String>, String> {
    let ext_refs: Vec<&str> = extensions.iter().map(String::as_str).collect();
    let Some(picked) = app
        .dialog()
        .file()
        .add_filter(&filter_name, &ext_refs)
        .set_file_name(&default_filename)
        .blocking_save_file()
    else {
        return Ok(None);
    };
    let target = picked.into_path().map_err(|e| e.to_string())?;
    std::fs::write(&target, content).map_err(|e| e.to_string())?;
    Ok(Some(target.display().to_string()))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            run_tens,
            open_tens,
            read_tens,
            list_folder,
            save_tens,
            export_text,
            print_window
        ])
        .run(tauri::generate_context!())
        .expect("error while running TensorForge");
}
