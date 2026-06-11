use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let path = match args.as_slice() {
        [_, cmd, path] if cmd == "run" => path,
        [_, path] if !path.starts_with('-') => path,
        _ => {
            eprintln!("TensorForge {} — symbolic tensor algebra for continuum mechanics", env!("CARGO_PKG_VERSION"));
            eprintln!();
            eprintln!("Usage: tensorforge run <file.tens>");
            return ExitCode::from(2);
        }
    };

    let src = match std::fs::read_to_string(path) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("error: cannot read `{path}`: {e}");
            return ExitCode::FAILURE;
        }
    };

    match tensorforge::run_source(&src) {
        Ok(outputs) => {
            for out in outputs {
                println!("[{}]", out.header);
                println!("{}", out.latex);
                println!();
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
