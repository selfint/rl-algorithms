use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use rand::Rng;

use neuron::layers::relu::ReLuLayer;
use neuron::network::FeedForwardNetwork;
use neuron::Layer;
use rand::prelude::ThreadRng;

#[derive(Debug, PartialEq, Clone)]
pub struct Agent {
    network: FeedForwardNetwork<ReLuLayer>,
}

impl Agent {
    pub fn new(dims: &[usize]) -> Self {
        assert!(!dims.is_empty());
        let layers: Vec<ReLuLayer> = (dims.iter().zip(dims.iter().skip(1)))
            .map(|(&layer, &next_layer)| ReLuLayer::new(next_layer, layer))
            .collect();

        Agent {
            network: FeedForwardNetwork::new(&layers),
        }
    }

    pub fn act(&self, state: &Array1<f32>) -> usize {
        self.network.predict(state).argmax().unwrap()
    }
}

pub struct NeuroEvolution {
    pub population: Vec<Agent>,
}

impl NeuroEvolution {
    pub fn new(agents: usize, network_dimensions: &[usize]) -> Self {
        Self {
            population: (0..agents)
                .map(|_| Agent::new(network_dimensions))
                .collect(),
        }
    }

    pub fn new_generation(&mut self, scores: &Vec<f32>) {}

    pub fn child(a1: &Agent, a2: &Agent) -> Agent {
        let mut rng = rand::thread_rng();

        let mut child_network = a1.network.clone();
        NeuroEvolution::crossover_weights(&mut child_network, a2, &mut rng);
        NeuroEvolution::crossover_biases(&mut child_network, a2, &mut rng);

        Agent {
            network: child_network,
        }
    }

    pub fn mutate(agent: &mut Agent) {
        let mut rng = rand::thread_rng();
        if rng.gen_bool(0.5) {
            NeuroEvolution::mutate_weight(agent);
        } else {
            NeuroEvolution::mutate_bias(agent);
        }
    }

    fn crossover_weights(
        child_network: &mut FeedForwardNetwork<ReLuLayer>,
        a2: &Agent,
        rng: &mut ThreadRng,
    ) {
        for (child_weights, a2_weights) in child_network
            .layers
            .iter_mut()
            .map(|l| l.get_weights_mut())
            .zip(a2.network.layers.iter().map(|l| l.get_weights()))
        {
            for i in 0..child_weights.nrows() {
                for j in 0..child_weights.ncols() {
                    if rng.gen_bool(0.5) {
                        child_weights[[i, j]] = a2_weights[[i, j]];
                    }
                }
            }
        }
    }

    fn crossover_biases(
        child_network: &mut FeedForwardNetwork<ReLuLayer>,
        a2: &Agent,
        rng: &mut ThreadRng,
    ) {
        for (child_biases, a2_biases) in child_network
            .layers
            .iter_mut()
            .map(|l| l.get_biases_mut())
            .zip(a2.network.layers.iter().map(|l| l.get_biases()))
        {
            for i in 0..child_biases.len() {
                if rng.gen_bool(0.5) {
                    child_biases[i] = a2_biases[i];
                }
            }
        }
    }

    fn mutate_weight(agent: &mut Agent) {
        let mut rng = rand::thread_rng();
        let layer = rng.gen_range(0..agent.network.layers.len() - 1);
        let layer_weights = agent.network.layers[layer].get_weights_mut();
        let node = rng.gen_range(0..layer_weights.nrows());
        let previous_node = rng.gen_range(0..layer_weights.ncols());
        let weight = layer_weights.get_mut([node, previous_node]).unwrap();

        *weight = rng.gen_range(-0.01..0.01);
    }

    fn mutate_bias(agent: &mut Agent) {
        let mut rng = rand::thread_rng();
        let layer = rng.gen_range(0..agent.network.layers.len() - 1);
        let layer_biases = agent.network.layers[layer].get_biases_mut();
        let node = rng.gen_range(0..layer_biases.len());
        let bias = layer_biases.get_mut(node).unwrap();

        *bias = rng.gen_range(-0.01..0.01);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new_generation() {
        let mut neuro_evolution = NeuroEvolution::new(10, &[2, 3, 1]);
        let first_generation = neuro_evolution.population.clone();
        assert_eq!(neuro_evolution.population.len(), 10);

        // score population
        let mut rng = rand::thread_rng();
        let scores = (0..10).map(|_| rng.gen_range(-2.0..10.0)).collect();
        neuro_evolution.new_generation(&scores);
    }

    #[test]
    fn test_mutate() {
        let mut a = Agent::new(&[2, 3, 1]);
        let network_weights_before: Vec<Array2<f32>> = a
            .network
            .layers
            .iter()
            .map(|l| l.get_weights().clone())
            .collect();
        let network_biases_before: Vec<Array1<f32>> = a
            .network
            .layers
            .iter()
            .map(|l| l.get_biases().clone())
            .collect();

        NeuroEvolution::mutate(&mut a);

        let network_weights_after: Vec<Array2<f32>> = a
            .network
            .layers
            .iter()
            .map(|l| l.get_weights().clone())
            .collect();
        let network_biases_after: Vec<Array1<f32>> = a
            .network
            .layers
            .iter()
            .map(|l| l.get_biases().clone())
            .collect();

        assert!(
            network_weights_before != network_weights_after
                || network_biases_before != network_biases_after
        );
    }

    #[test]
    fn test_child() {
        let a1 = Agent::new(&[2, 3, 1]);
        let a2 = Agent::new(&[2, 3, 1]);
        let child = NeuroEvolution::child(&a1, &a2);

        assert_ne!(child.network, a1.network);
        assert_ne!(child.network, a2.network);
        assert_eq!(child.network.shape(), a1.network.shape());
    }

    #[test]
    fn test_agent_action() {
        let agent = Agent::new(&[2, 3, 1]);
        let state = arr1(&[0., 1.]);
        let actions = 2;
        let agent_action = agent.act(&state);

        assert!(agent_action < actions);
    }
}
