use std::process::ExitCode;

const USAGE: &str = "\
TensorForge — rigorous symbolic tensor algebra for continuum mechanics

Usage:
  tensorforge run <file.tens>     parse, derive, and print LaTeX results
  tensorforge --version | -V      print the version
  tensorforge --help    | -h      print this help

Examples:
  tensorforge run examples/hill_cr.tens
  tensorforge run examples/neo_hookean.tens";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let argv: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();
    let path = match argv.as_slice() {
        ["--version" | "-V"] => {
            println!("tensorforge {}", env!("CARGO_PKG_VERSION"));
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
