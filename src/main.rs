use std::process::ExitCode;

const USAGE: &str = "\
Tensbook — a symbolic math notebook: tensors, calculus, ODEs, and plots

Usage:
  tensbook run <file.tens>     parse, derive, and print LaTeX results
  tensbook --version | -V      print the version
  tensbook --help    | -h      print this help

Examples:
  tensbook run examples/start.tens";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let argv: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();
    let path = match argv.as_slice() {
        ["--version" | "-V"] => {
            println!("tensbook {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        ["--help" | "-h"] => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        ["run", path] => path,
        [path] if !path.starts_with('-') => path,
        _ => {
            eprintln!("{USAGE}");
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

    match tensbook::run_source(&src) {
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
