//! General Model Orchestration Layer
//!
//! This module defines the core trait for hardware-agnostic model orchestration.
//! It allows the MoFA framework to seamlessly switch between different inference
//! backends (e.g., Apple MLX, HuggingFace Candle, ONNX) depending on the
//! underlying operating system and available hardware.

use anyhow::Result;
use futures::Stream;
use std::pin::Pin;

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

pub trait HardwareOracle: Send + Sync {
    /// Returns the available memory in bytes
    fn get_available_memory_bytes(&self) -> u64;

    /// Returns the total memory in bytes
    fn get_total_memory_bytes(&self) -> u64;
}

pub struct SysinfoOracle {
    sys: std::sync::Mutex<sysinfo::System>,
}

impl Default for SysinfoOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl SysinfoOracle {
    pub fn new() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        Self {
            sys: std::sync::Mutex::new(sys),
        }
    }
}

impl HardwareOracle for SysinfoOracle {
    fn get_available_memory_bytes(&self) -> u64 {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        sys.available_memory()
    }

    fn get_total_memory_bytes(&self) -> u64 {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        sys.total_memory()
    }
}

pub trait ModelOrchestrator: Send + Sync {
    fn initialize(&mut self) -> Result<()>;

    /// Register a model with its expected memory footprint (in bytes)
    fn register_model(&mut self, model_id: &str, footprint_bytes: u64) -> Result<()>;

    fn load_model(&mut self, model_id: &str) -> Result<()>;

    fn unload_model(&mut self, model_id: &str) -> Result<()>;

    fn is_model_loaded(&self, model_id: &str) -> bool;

    fn generate(&self, model_id: &str, prompt: &str) -> Result<TokenStream>;
}

pub struct MockOrchestratorState {
    /// Maps model_id -> footprint in bytes
    registered_models: std::collections::HashMap<String, u64>,
    /// Tracks models currently loaded in memory
    loaded_models: std::collections::HashSet<String>,
    /// LRU queue for tracking the order in which models were most recently used/loaded
    lru_queue: std::collections::VecDeque<String>,
    /// Total available mock memory budget (if overriding the hardware oracle)
    mock_memory_budget_bytes: u64,
}

pub struct MockOrchestrator {
    state: std::sync::Mutex<MockOrchestratorState>,
}

impl Default for MockOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl MockOrchestrator {
    pub fn new() -> Self {
        Self {
            state: std::sync::Mutex::new(MockOrchestratorState {
                registered_models: std::collections::HashMap::new(),
                loaded_models: std::collections::HashSet::new(),
                lru_queue: std::collections::VecDeque::new(),
                // Mock a 16GB system
                mock_memory_budget_bytes: 16_000_000_000,
            }),
        }
    }

    /// Helper to get current memory usage
    fn current_memory_usage(&self) -> u64 {
        let state = self.state.lock().unwrap();
        state
            .loaded_models
            .iter()
            .filter_map(|model_id| state.registered_models.get(model_id))
            .sum()
    }
}

impl ModelOrchestrator for MockOrchestrator {
    fn initialize(&mut self) -> Result<()> {
        let state = self.state.lock().unwrap();
        println!(
            "[MockOrchestrator] Initialized with {} bytes budget.",
            state.mock_memory_budget_bytes
        );
        Ok(())
    }

    fn register_model(&mut self, model_id: &str, footprint_bytes: u64) -> Result<()> {
        println!(
            "[MockOrchestrator] Registered model {} with footprint {} bytes.",
            model_id, footprint_bytes
        );
        let mut state = self.state.lock().unwrap();
        state
            .registered_models
            .insert(model_id.to_string(), footprint_bytes);
        Ok(())
    }

    fn load_model(&mut self, model_id: &str) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let footprint = state.registered_models.get(model_id).copied().unwrap_or(0);

        println!(
            "[MockOrchestrator] Attempting to load model: {} (Requires: {} bytes)",
            model_id, footprint
        );

        // Admission Gate & Eviction Loop
        loop {
            let current_usage: u64 = state
                .loaded_models
                .iter()
                .filter_map(|id| state.registered_models.get(id))
                .sum();

            if current_usage + footprint <= state.mock_memory_budget_bytes {
                break;
            }

            if let Some(evict_candidate) = state.lru_queue.pop_front() {
                println!(
                    "[Admission Gate] Budget exceeded! Evicting LRU model: {}",
                    evict_candidate
                );
                state.loaded_models.remove(&evict_candidate);
                // Also remove any remaining instances of it from the queue just in case
                state.lru_queue.retain(|id| id != &evict_candidate);
            } else {
                return Err(anyhow::anyhow!(
                    "Admission Gate blocked load: Model {} footprint ({}) exceeds total system budget ({})",
                    model_id,
                    footprint,
                    state.mock_memory_budget_bytes
                ));
            }
        }

        state.loaded_models.insert(model_id.to_string());
        state.lru_queue.retain(|id| id != model_id); // Remove if exists
        state.lru_queue.push_back(model_id.to_string()); // Mark as most recently used

        // Re-calculate usage for printing
        let current_usage: u64 = state
            .loaded_models
            .iter()
            .filter_map(|id| state.registered_models.get(id))
            .sum();

        println!(
            "[MockOrchestrator] Successfully loaded model: {}. Current usage: {} / {}",
            model_id, current_usage, state.mock_memory_budget_bytes
        );
        Ok(())
    }

    fn unload_model(&mut self, model_id: &str) -> Result<()> {
        println!("[MockOrchestrator] Unloading model: {}", model_id);
        let mut state = self.state.lock().unwrap();
        state.loaded_models.remove(model_id);
        state.lru_queue.retain(|id| id != model_id);
        Ok(())
    }

    fn is_model_loaded(&self, model_id: &str) -> bool {
        let state = self.state.lock().unwrap();
        state.loaded_models.contains(model_id)
    }

    fn generate(&self, model_id: &str, prompt: &str) -> Result<TokenStream> {
        if !self.is_model_loaded(model_id) {
            return Err(anyhow::anyhow!("Model {} is not loaded.", model_id));
        }

        let mut state = self.state.lock().unwrap();
        // Update LRU on use
        state.lru_queue.retain(|id| id != model_id);
        state.lru_queue.push_back(model_id.to_string());

        println!(
            "[MockOrchestrator] Generating response for prompt: '{}' using model: {}",
            prompt, model_id
        );

        // Return a dummy stream with a single token for the mock
        let stream = futures::stream::iter(vec![Ok("Mock response generated.".to_string())]);
        Ok(Box::pin(stream))
    }
}
