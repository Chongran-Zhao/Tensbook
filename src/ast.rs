//! Syntactic AST produced by the parser.
//!
//! This layer knows nothing about mathematics: `F.T * F` is just a
//! `Binary(Mul, Field(F, "T"), F)`. The interpreter lowers it into the
//! semantic representations in [`crate::symbolic`] and [`crate::tensor`].

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `name = expr`
    Assign {
        name: String,
        expr: Expr,
        line: usize,
    },
    /// A bare expression statement, e.g. `display(C, mode=symbol)`.
    Expr(Expr, usize),
}

impl Stmt {
    pub fn line(&self) -> usize {
        match self {
            Stmt::Assign { line, .. } => *line,
            Stmt::Expr(_, line) => *line,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Str(String),
    Bool(bool),
    Ident(String),
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Property access, e.g. `F.T`.
    Field {
        target: Box<Expr>,
        name: String,
    },
    /// `callee(args..., key=value...)`
    Call {
        callee: String,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },
    /// Set element access, e.g. `lambda[a]` or `N[1]`. The index is an
    /// abstract index name (Ident) or a concrete position (Num).
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// Outer product `A & B`.
    Outer,
    /// Double contraction `A : B`.
    Ddot,
}
