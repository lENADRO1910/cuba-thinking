mod engine;
mod server;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize Strict Structured Logging (SRE Pillar)
    tracing_subscriber::fmt()
        //.with_env_filter(EnvFilter::from_default_env()) // (Needs appropriate dependency setup)
        .with_writer(std::io::stderr) // STDERR is crucial so STDOUT is reserved for MCP JSON
        .init();

    tracing::info!("Booting Antigravity SOTA Deep Reasoning Engine (Rust v0.1.0)...");

    // 2. Initialize the MCP Protocol Server 
    // This replaces the entire Python layer.
    let mcp_server = std::sync::Arc::new(server::McpServer::new());
    
    // 3. Block on the execution loop
    mcp_server.run().await?;
    
    Ok(())
}
