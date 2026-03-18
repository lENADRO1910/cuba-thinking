// src/engine/thought_graph.rs
//
// R9: Graph-of-Thought (GoT) Lite — DAG + Topology Analysis
//
// Extends the flat MCTS tree into a Directed Acyclic Graph for
// tracking thought dependencies, revision chains, and convergence.
// Based on Besta et al. 2024 (ETH Zurich) "Graph of Thoughts".
//
// Features:
// - DAG edge tracking (thought A → thought B dependencies)
// - Convergence detection (multiple paths leading to same conclusion)
// - Revision chain tracking (which thoughts revised which)
// - Depth analysis (longest path = reasoning depth)

use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
// Note: HashSet is used by detect_cycles (Tarjan), VecDeque by max_depth (Kahn).

/// A lightweight DAG for tracking thought relationships.
#[derive(Debug, Clone, Serialize)]
pub struct ThoughtGraph {
    /// Adjacency list: thought_id → list of dependent thought_ids.
    edges: HashMap<usize, Vec<usize>>,
    /// Reverse adjacency: thought_id → list of parent thought_ids.
    reverse_edges: HashMap<usize, Vec<usize>>,
    /// Revision records: (revised_thought_id, revising_thought_id).
    revisions: Vec<(usize, usize)>,
    /// Total node count.
    pub node_count: usize,
}

impl ThoughtGraph {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            revisions: Vec::new(),
            node_count: 0,
        }
    }

    /// Add a thought node to the graph.
    pub fn add_node(&mut self, thought_id: usize) {
        self.edges.entry(thought_id).or_default();
        self.reverse_edges.entry(thought_id).or_default();
        self.node_count = self.node_count.max(thought_id + 1);
    }

    /// Add a dependency edge: source_thought → dependent_thought.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.entry(from).or_default().push(to);
        self.reverse_edges.entry(to).or_default().push(from);
    }

    /// Record a revision: thought `reviser` revises thought `original`.
    /// Reserved for multi-thought chain analysis.
    #[allow(dead_code)]
    pub fn add_revision(&mut self, original: usize, reviser: usize) {
        self.revisions.push((original, reviser));
        self.add_edge(original, reviser);
    }

    /// Compute the longest path in the DAG (reasoning depth).
    ///
    /// V7: Uses Kahn's topological sort + dynamic programming.
    /// Previous BFS approach incorrectly skipped longer paths to
    /// already-visited nodes in DAGs with convergence.
    ///
    /// Complexity: O(V + E) — single pass.
    pub fn max_depth(&self) -> usize {
        if self.node_count == 0 {
            return 0;
        }

        // DP table: longest path ending at each node (1-indexed depth).
        let mut dist = vec![0usize; self.node_count];

        // In-degree for Kahn's algorithm.
        let mut in_degree = vec![0usize; self.node_count];
        for children in self.edges.values() {
            for &child in children {
                if child < self.node_count {
                    in_degree[child] += 1;
                }
            }
        }

        // Initialize roots (in-degree 0) with depth 1.
        let mut queue = VecDeque::new();
        for id in 0..self.node_count {
            if in_degree[id] == 0 {
                dist[id] = 1;
                queue.push_back(id);
            }
        }

        // Process in topological order (Kahn's algorithm).
        // For each node, propagate: dist[child] = max(dist[child], dist[node] + 1).
        while let Some(node) = queue.pop_front() {
            if let Some(children) = self.edges.get(&node) {
                for &child in children {
                    if child < self.node_count {
                        dist[child] = dist[child].max(dist[node] + 1);
                        in_degree[child] -= 1;
                        if in_degree[child] == 0 {
                            queue.push_back(child);
                        }
                    }
                }
            }
        }

        // Handle orphan nodes (no edges, in_degree=0, dist=1 from init).
        // Handle disconnected components (roots already seeded).
        dist.iter().copied().max().unwrap_or(0)
    }

    /// Detect convergence: nodes with multiple incoming edges.
    /// These represent conclusions reached from multiple reasoning paths.
    pub fn convergence_points(&self) -> Vec<usize> {
        self.reverse_edges
            .iter()
            .filter(|(_, parents)| parents.len() > 1)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Count revision chains (how many times thoughts were revised).
    pub fn revision_count(&self) -> usize {
        self.revisions.len()
    }

    /// Get orphan nodes (no edges in or out, isolated thoughts).
    pub fn orphans(&self) -> Vec<usize> {
        (0..self.node_count)
            .filter(|id| {
                let no_out = self.edges.get(id).is_none_or(|e| e.is_empty());
                let no_in = self.reverse_edges.get(id).is_none_or(|e| e.is_empty());
                no_out && no_in
            })
            .collect()
    }

    /// V6-2: Detect circular reasoning using Tarjan's SCC algorithm.
    ///
    /// Finds all Strongly Connected Components with |V| > 1.
    /// A cycle in the dependency graph means the LLM's reasoning is
    /// circular (petitio principii): "X is true because Y" + "Y is true because X".
    ///
    /// Complexity: O(V + E) — single-pass DFS (Tarjan, 1972).
    pub fn detect_cycles(&self) -> Vec<Vec<usize>> {
        let mut index_counter: usize = 0;
        let mut stack: Vec<usize> = Vec::new();
        let mut on_stack: HashSet<usize> = HashSet::new();
        let mut indices: HashMap<usize, usize> = HashMap::new();
        let mut lowlinks: HashMap<usize, usize> = HashMap::new();
        let mut sccs: Vec<Vec<usize>> = Vec::new();

        // Iterative Tarjan to avoid stack overflow on deep graphs.
        for start_node in 0..self.node_count {
            if indices.contains_key(&start_node) {
                continue;
            }

            // DFS work stack: (node, child_index, is_root_call)
            let mut work: Vec<(usize, usize)> = vec![(start_node, 0)];
            indices.insert(start_node, index_counter);
            lowlinks.insert(start_node, index_counter);
            index_counter += 1;
            stack.push(start_node);
            on_stack.insert(start_node);

            while let Some((node, child_idx)) = work.last_mut() {
                let children: Vec<usize> = self.edges.get(node).cloned().unwrap_or_default();

                if *child_idx < children.len() {
                    let child = children[*child_idx];
                    *child_idx += 1;

                    // Clippy suggests Entry API, but this if-else if pattern
                    // (indices check + on_stack check) doesn't map to Entry cleanly.
                    #[allow(clippy::map_entry)]
                    if !indices.contains_key(&child) {
                        // Tree edge: descend
                        indices.insert(child, index_counter);
                        lowlinks.insert(child, index_counter);
                        index_counter += 1;
                        stack.push(child);
                        on_stack.insert(child);
                        work.push((child, 0));
                    } else if on_stack.contains(&child) {
                        // Back edge: cycle detected — update lowlink
                        let node_ll = *lowlinks.get(node).unwrap();
                        let child_idx_val = *indices.get(&child).unwrap();
                        lowlinks.insert(*node, node_ll.min(child_idx_val));
                    }
                } else {
                    // All children processed — check if node is SCC root
                    let node_val = *node;
                    let node_idx = *indices.get(&node_val).unwrap();
                    let node_ll = *lowlinks.get(&node_val).unwrap();

                    if node_ll == node_idx {
                        // SCC root: pop stack until we reach this node
                        let mut scc = Vec::new();
                        while let Some(w) = stack.pop() {
                            on_stack.remove(&w);
                            scc.push(w);
                            if w == node_val {
                                break;
                            }
                        }
                        // Only report cycles (SCC with >1 node)
                        if scc.len() > 1 {
                            scc.sort_unstable();
                            sccs.push(scc);
                        }
                    }

                    // Propagate lowlink to parent
                    work.pop();
                    if let Some((parent, _)) = work.last() {
                        let parent_ll = *lowlinks.get(parent).unwrap();
                        lowlinks.insert(*parent, parent_ll.min(node_ll));
                    }
                }
            }
        }

        sccs
    }

    /// Check if the graph has any circular reasoning.
    #[allow(dead_code)]
    pub fn has_cycles(&self) -> bool {
        !self.detect_cycles().is_empty()
    }

    /// V9: Effective depth discounted by Tarjan cycle ratio.
    ///
    /// Raw depth from Kahn's algorithm counts all nodes equally.
    /// Circular reasoning (petitio principii) inflates depth without
    /// adding genuine reasoning progress. This metric penalizes depth
    /// proportionally to the fraction of nodes involved in cycles.
    ///
    /// Formula: depth_eff = depth_raw × (1 - cycle_nodes / total_nodes)
    ///
    /// Based on DoT (Depth of Thought, Zhang et al. 2024).
    #[allow(dead_code)]
    pub fn effective_depth(&self) -> f64 {
        let raw = self.max_depth() as f64;
        if self.node_count == 0 {
            return 0.0;
        }
        let cycles = self.detect_cycles();
        let cycle_nodes: usize = cycles.iter().map(|c| c.len()).sum();
        let penalty = 1.0 - (cycle_nodes as f64 / self.node_count as f64);
        raw * penalty
    }

    /// Generate topology summary for the formatter.
    pub fn topology_summary(&self) -> TopologySummary {
        let cycles = self.detect_cycles();
        TopologySummary {
            total_nodes: self.node_count,
            total_edges: self.edges.values().map(|v| v.len()).sum(),
            max_depth: self.max_depth(),
            convergence_points: self.convergence_points().len(),
            revision_count: self.revision_count(),
            orphan_count: self.orphans().len(),
            cycle_count: cycles.len(),
        }
    }

    /// V5-1: Prune graph nodes and edges beyond `target_thought`.
    /// Removes dead branch state when MCTS rejects a thought to prevent
    /// hallucinated dependencies from influencing future topology analysis.
    pub fn prune_after(&mut self, target_thought: usize) {
        // Remove nodes beyond target
        self.edges.retain(|&k, _| k <= target_thought);
        self.reverse_edges.retain(|&k, _| k <= target_thought);

        // Remove edges pointing to pruned nodes
        for edges in self.edges.values_mut() {
            edges.retain(|&to| to <= target_thought);
        }
        for edges in self.reverse_edges.values_mut() {
            edges.retain(|&from| from <= target_thought);
        }

        // Remove stale revisions
        self.revisions
            .retain(|&(a, b)| a <= target_thought && b <= target_thought);

        // Update node count
        self.node_count = target_thought + 1;
    }
}

/// Compact topology summary for output formatting.
#[derive(Debug, Clone, Serialize)]
pub struct TopologySummary {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub max_depth: usize,
    pub convergence_points: usize,
    pub revision_count: usize,
    pub orphan_count: usize,
    /// V6-2: Number of circular reasoning cycles (SCC with |V|>1).
    pub cycle_count: usize,
}

impl TopologySummary {
    /// Compact display string.
    pub fn display(&self) -> String {
        let cycle_warn = if self.cycle_count > 0 {
            format!(" ⚠️ {} circular reasoning cycles!", self.cycle_count)
        } else {
            String::new()
        };
        format!(
            "🌐 GoT: {} nodes, {} edges, depth {}, {} convergence, {} revisions{}",
            self.total_nodes,
            self.total_edges,
            self.max_depth,
            self.convergence_points,
            self.revision_count,
            cycle_warn,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_graph_depth() {
        let mut graph = ThoughtGraph::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        assert_eq!(graph.max_depth(), 5);
    }

    #[test]
    fn test_convergence_detection() {
        let mut graph = ThoughtGraph::new();
        for i in 0..4 {
            graph.add_node(i);
        }
        // Two paths converge at node 3
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);
        let convergence = graph.convergence_points();
        assert!(convergence.contains(&3));
    }

    #[test]
    fn test_revision_tracking() {
        let mut graph = ThoughtGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_revision(0, 1); // Thought 1 revises thought 0
        assert_eq!(graph.revision_count(), 1);
    }

    #[test]
    fn test_orphan_detection() {
        let mut graph = ThoughtGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        graph.add_edge(0, 1);
        // Node 2 is orphan (no edges)
        let orphans = graph.orphans();
        assert!(orphans.contains(&2));
        assert!(!orphans.contains(&0));
    }

    #[test]
    fn test_empty_graph() {
        let graph = ThoughtGraph::new();
        assert_eq!(graph.max_depth(), 0);
        assert!(graph.convergence_points().is_empty());
    }

    // ─── V7: DP Topological Sort Depth Tests ──────

    #[test]
    fn test_convergent_dag_depth() {
        // P3-A regression test: BFS incorrectly returned 3 for this graph.
        // The longest path is 0→2→1→3 = depth 4.
        //
        //    0 → 1 → 3
        //    0 → 2 → 1
        //
        // BFS visits node 1 at depth 2 (via 0→1), marks visited.
        // Path 0→2→1→3 has depth 4 but node 1 already visited → skipped.
        let mut graph = ThoughtGraph::new();
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1); // 0 → 1
        graph.add_edge(0, 2); // 0 → 2
        graph.add_edge(1, 3); // 1 → 3
        graph.add_edge(2, 1); // 2 → 1 (convergence: node 1 reachable via 0→1 AND 0→2→1)
        assert_eq!(
            graph.max_depth(),
            4,
            "Longest path 0→2→1→3 should give depth 4, not 3"
        );
    }

    #[test]
    fn test_diamond_dag_depth() {
        // Diamond: 0 → {1, 2} → 3
        // All paths have same depth = 3
        let mut graph = ThoughtGraph::new();
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);
        assert_eq!(graph.max_depth(), 3, "Diamond DAG depth should be 3");
    }

    // ─── V6-2: Tarjan SCC Cycle Detection Tests ──────

    #[test]
    fn test_dag_no_cycles() {
        let mut graph = ThoughtGraph::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        assert!(
            graph.detect_cycles().is_empty(),
            "Linear DAG should have no cycles"
        );
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_simple_cycle() {
        let mut graph = ThoughtGraph::new();
        for i in 0..3 {
            graph.add_node(i);
        }
        // Create cycle: 0 → 1 → 2 → 0
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        let cycles = graph.detect_cycles();
        assert_eq!(cycles.len(), 1, "Should detect 1 cycle");
        assert_eq!(cycles[0].len(), 3, "Cycle should have 3 nodes");
        assert!(graph.has_cycles());
    }

    #[test]
    fn test_petitio_principii() {
        // "X is true because Y" + "Y is true because X"
        let mut graph = ThoughtGraph::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        // Linear chain: 0 → 1 → 2 → 3 → 4
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        // Back-edge creates circular reasoning: 4 → 1
        graph.add_edge(4, 1);
        let cycles = graph.detect_cycles();
        assert_eq!(cycles.len(), 1, "Should detect circular reasoning");
        // Cycle includes nodes 1, 2, 3, 4
        assert!(cycles[0].len() >= 2, "Cycle should have multiple nodes");
    }

    #[test]
    fn test_multiple_cycles() {
        let mut graph = ThoughtGraph::new();
        for i in 0..6 {
            graph.add_node(i);
        }
        // Cycle 1: 0 → 1 → 0
        graph.add_edge(0, 1);
        graph.add_edge(1, 0);
        // Cycle 2: 3 → 4 → 5 → 3
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 3);
        // Bridge (no cycle): 2 → 3
        graph.add_edge(2, 3);
        let cycles = graph.detect_cycles();
        assert_eq!(cycles.len(), 2, "Should detect 2 separate cycles");
    }

    #[test]
    fn test_topology_summary_includes_cycles() {
        let mut graph = ThoughtGraph::new();
        for i in 0..3 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        let summary = graph.topology_summary();
        assert_eq!(summary.cycle_count, 1);
        assert!(summary.display().contains("circular reasoning"));
    }
}
