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
        block: usize,
    },
    /// Component assignment into a declared tensor: `F[1][1] = expr`
    /// (one index group per tensor order).
    AssignComponent {
        name: String,
        indices: Vec<Expr>,
        expr: Expr,
        line: usize,
        block: usize,
    },
    /// Destructuring assignment `[a, b] = Spec_Decomp(C)`.
    AssignPair {
        first: String,
        second: String,
        expr: Expr,
        line: usize,
        block: usize,
    },
    /// A bare expression statement, e.g. `C.show(matrix)`.
    Expr(Expr, usize, usize),
    /// A row of output calls, e.g. `[I1.show() I2.show()]`.
    OutputRow {
        exprs: Vec<Expr>,
        line: usize,
        block: usize,
    },
}

impl Stmt {
    pub fn line(&self) -> usize {
        match self {
            Stmt::Assign { line, .. } => *line,
            Stmt::AssignComponent { line, .. } => *line,
            Stmt::AssignPair { line, .. } => *line,
            Stmt::Expr(_, line, _) => *line,
            Stmt::OutputRow { line, .. } => *line,
        }
    }

    pub fn block(&self) -> usize {
        match self {
            Stmt::Assign { block, .. } => *block,
            Stmt::AssignComponent { block, .. } => *block,
            Stmt::AssignPair { block, .. } => *block,
            Stmt::Expr(_, _, block) => *block,
            Stmt::OutputRow { block, .. } => *block,
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
    /// Method call, e.g. `C.show(matrix)`.
    MethodCall {
        target: Box<Expr>,
        method: String,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
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
