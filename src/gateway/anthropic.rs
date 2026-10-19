//! TODO: anthropic dialect (stub).
use axum::{extract::{Request, State}, response::Response, http::StatusCode, response::IntoResponse};
use super::Shared;

pub async fn handle(State(_st): State<Shared>, _req: Request) -> Response {
    (StatusCode::NOT_IMPLEMENTED, "anthropic: not implemented").into_response()
}

pub async fn count_tokens(State(_st): State<Shared>, _req: Request) -> Response {
    (StatusCode::NOT_IMPLEMENTED, "not implemented").into_response()
}
