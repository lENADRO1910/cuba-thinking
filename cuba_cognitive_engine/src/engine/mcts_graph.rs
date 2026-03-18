use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A single step in the thought process. 
/// 
/// We use an Index-Based Graph (Data-Oriented Design) instead of Rc<RefCell> 
/// to mathematically eliminate runtime pointer overhead and guarantee O(1) memory cleanup 
/// at the end of the thought generation when the Arena is dropped.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ThoughtNode<'a> {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub children: BumpVec<'a, usize>,
    
    /// The string delta or thought generated at this step.
    /// In a fully mmap-backed ggml integration, this could just be a token slice.
    pub content: &'a str,
    
    // MCTS Statistics
    pub visits: AtomicUsize,
    pub q_value: f64,             // Mean reward for this node (Exploitation)
    pub prior_probability: f64,   // Policy probability from the Base Model (Exploration)
    pub variance: f64,            // Statistical variance of PRM scores (Adaptive UCT)
    pub correlation_lambda: f64,  // Hoeffding discount factor for Agent Team state sharing
}

/// The Orchestrator holding all nodes in a flat vector for CPU Cache Locality.
#[allow(dead_code)]
pub struct MctsGraph<'a> {
    /// The memory arena. No node survives outside this arena.
    pub arena: &'a Bump,
    
    /// Flat storage of all nodes. The index in this Vec is the `NodeId`.
    pub nodes: BumpVec<'a, ThoughtNode<'a>>,
    
    /// Exploration hyperparameter (c_puct). Controls Depth vs Breadth.
    pub c_puct: f64,
}

#[allow(dead_code)]
impl<'a> MctsGraph<'a> {
    pub fn new(arena: &'a Bump, c_puct: f64) -> Self {
        Self {
            arena,
            nodes: BumpVec::new_in(arena),
            c_puct,
        }
    }

    /// PUCT Formula + Adaptive UCT based on statistical node variance.
    /// P1-1: Safe bounds check — returns Result instead of panicking on OOB.
    pub fn calculate_puct(&self, node_id: usize) -> Result<f64, &'static str> {
        let node = self.nodes.get(node_id).ok_or("MCTS: node_id out of bounds")?;
        let (parent_visits, parent_variance) = match node.parent_id {
            Some(pid) => {
                let p = self.nodes.get(pid).ok_or("MCTS: parent_id out of bounds")?;
                (p.visits.load(Ordering::Relaxed) as f64, p.variance)
            },
            None => (1.0, 0.0),
        };
        
        // Adaptive c_puct logic: High variance = High exploration.
        let dynamic_c = if parent_variance > 0.3 {
            self.c_puct * 2.0 // Widen the search tree when PRM is uncertain
        } else if parent_visits > 5.0 && parent_variance < 0.1 {
            self.c_puct * 0.5 // Collapse the search tree when consensus is reached
        } else {
            self.c_puct
        };
        
        // Apply correlation discount for multi-agent shared states
        let q = node.q_value * node.correlation_lambda;
        let p = node.prior_probability;
        let n = node.visits.load(Ordering::Relaxed) as f64;
        
        Ok(q + dynamic_c * p * (parent_visits.sqrt() / (1.0 + n)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_arena_allocation_and_puct_math() {
        // 1. Initialize the Arena for O(1) memory cleanup
        let arena = Bump::new();
        let mut mcts = MctsGraph::new(&arena, 1.25); // c_puct = 1.25

        // 2. Create a Root Node 
        let root = ThoughtNode {
            id: 0,
            parent_id: None,
            children: BumpVec::new_in(&arena),
            content: "Root Thought",
            visits: AtomicUsize::new(10), // Visited 10 times
            q_value: 0.5,
            prior_probability: 1.0,
            variance: 0.2, // Keeps c_puct standard in test
            correlation_lambda: 1.0,
        };
        mcts.nodes.push(root);

        // 3. Create a Child Node
        let child = ThoughtNode {
            id: 1,
            parent_id: Some(0),
            children: BumpVec::new_in(&arena),
            content: "First Logical Step",
            visits: AtomicUsize::new(2), // Child visited 2 times
            q_value: 0.8,              // Good reward
            prior_probability: 0.4,    // Medium probability from Base LLM
            variance: 0.0,
            correlation_lambda: 1.0,
        };
        mcts.nodes.push(child);
        
        // Link child to root
        mcts.nodes[0].children.push(1);

        // 4. Calculate PUCT for the child
        // Formula: Q + c * P * sqrt(N_parent) / (1 + N_node)
        // Q = 0.8
        // c = 1.25
        // P = 0.4
        // N_parent = 10 -> sqrt(10) ≈ 3.162277
        // N_node = 2 -> 1 + 2 = 3
        // Result: 0.8 + 1.25 * 0.4 * 3.162277 / 3
        // Result: 0.8 + 0.5 * 3.162277 / 3
        // Result: 0.8 + 1.581138 / 3
        // Result: 0.8 + 0.527046 ≈ 1.327
        
        let puct_score = mcts.calculate_puct(1).expect("PUCT should succeed for valid node");
        
        // Use an epsilon for float comparison
        assert!((puct_score - 1.327046).abs() < 0.001, "PUCT mathematical failure: {}", puct_score);

        // P1-1: Verify OOB returns error instead of panic
        assert!(mcts.calculate_puct(999).is_err(), "OOB node_id should return Err");

        // 5. When `arena` goes out of scope here, all memory is instantly reclaimed (O(1)).
        // No Iterative dropping, no Garbage Collection overhead.
    }
}
