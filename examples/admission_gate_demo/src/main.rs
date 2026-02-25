use anyhow::Result;
use mofa_foundation::orchestrator::{MockOrchestrator, ModelOrchestrator};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== MoFA Admission Gate & Memory Oracle Demo ===");
    println!("Simulating a local Mac with 16GB of Unified Memory.");
    
    // 1. Initialize our Memory-Aware Orchestrator
    let mut orchestrator = MockOrchestrator::new();
    orchestrator.initialize()?;

    // 2. Register Model Footprints 
    // Agent A needs a large generic LLM (8GB)
    orchestrator.register_model("qwen-7b-int4", 8_000_000_000)?;
    
    // Agent B needs a specialized coding model (6GB)
    orchestrator.register_model("deepseek-coder-6.7b", 6_000_000_000)?;
    
    // Agent C needs a TTS audio model (4GB) 
    orchestrator.register_model("gpt-sovits-v2", 4_000_000_000)?;

    println!("\n--- Step 1: Agent A (General Chat) executed ---");
    // Load Qwen (Takes 8GB of 16GB)
    orchestrator.load_model("qwen-7b-int4")?;
    
    println!("\n--- Step 2: Agent B (Code Generation) executed ---");
    // Load DeepSeek (Takes 6GB, total is now 14GB of 16GB)
    orchestrator.load_model("deepseek-coder-6.7b")?;
    
    println!("\n--- Step 3: Agent C (Audio Response) executed ---");
    // Needs 4GB for TTS, but only 2GB is left!
    // The Admission Gate will block the load, trigger the LRU Eviction queue,
    // and dynamically unload the idle Qwen model to make room!
    orchestrator.load_model("gpt-sovits-v2")?;

    println!("\n=== Orchestration Completed Successfully! ===");
    println!("No OOM crashes occurred because the Admission Gate actively managed VRAM.");

    Ok(())
}
