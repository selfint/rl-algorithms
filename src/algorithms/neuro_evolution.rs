use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use neuron::layers::FullyConnected;

struct Agent {
    network: FullyConnected,
}

impl Agent {
    fn new(dims: &[usize]) -> Self {
        assert!(!dims.is_empty());

        Agent {
            network: FullyConnected::from(dims),
        }
    }

    fn act(&self, state: &[f32]) -> usize {
        self.network.predict(state).argmax().unwrap()
    }
}

struct NeuroEvolution;

impl NeuroEvolution {
    fn crossover(a1: &Agent, a2: &Agent) -> Agent {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_action() {
        let agent = Agent::new(&[2, 3, 1]);
        let state = [0., 1.];
        let actions = 2;
        let agent_action = agent.act(&state);

        assert!(agent_action >= 0);
        assert!(agent_action < actions);
    }

    #[test]
    fn test_crossover() {}
}
