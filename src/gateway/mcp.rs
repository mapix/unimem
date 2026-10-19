//! TODO: mcp dialect (stub).
use axum::{extract::{Request, State}, response::Response, http::StatusCode, response::IntoResponse};
use super::Shared;

pub async fn handle(State(_st): State<Shared>, _req: Request) -> Response {
    (StatusCode::NOT_IMPLEMENTED, "mcp: not implemented").into_response()
}
