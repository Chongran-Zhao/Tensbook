//! Regression tests pinning parser/lexer diagnostics: the message wording is
//! user-facing (the UI shows it verbatim) and the line number drives the
//! editor's error placement, so both are part of observable behavior.

use tensbook::run_source;

fn error_of(src: &str) -> tensbook::error::Error {
    run_source(src).unwrap_err()
}

#[test]
fn unterminated_string_reports_line() {
    let err = error_of("a = Scalar(\"x");
    assert!(err.message.contains("unterminated string literal"));
    assert_eq!(err.line, Some(1));
}

#[test]
fn unexpected_character_names_the_character() {
    let err = error_of("a ? 3");
    assert!(err.message.contains("unexpected character `?`"));
}

#[test]
fn unterminated_display_row() {
    let err = error_of("[Tr(3).show()");
    assert!(err.message.contains("unterminated display row"));
}

#[test]
fn empty_display_row() {
    let err = error_of("[]");
    assert!(err.message.contains("empty display row"));
}

#[test]
fn unexpected_token_in_expression() {
    let err = error_of("x = )");
    assert!(err.message.contains("unexpected token"));
}

#[test]
fn positional_after_keyword_argument() {
    let err = error_of("A = Tensor(\"\\bm A\", order=2, 3)");
    assert!(err
        .message
        .contains("positional argument after keyword argument"));
}

#[test]
fn list_literal_outside_plot_target() {
    let err = error_of("x = Var(\"x\")\ny = [sin(x), cos(x)]");
    assert!(err
        .message
        .contains("list literals are only valid as a `.plot(...)` target"));
    assert_eq!(err.line, Some(2));
}

#[test]
fn errors_on_later_lines_keep_their_line_number() {
    let err = error_of("a = Scalar(\"a\")\nb = Scalar(\"b\")\nc = Scalar(\"c");
    assert!(err.message.contains("unterminated string literal"));
    assert_eq!(err.line, Some(3));
}
