use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use rand::Rng;

use neuron::layers::FullyConnected;

struct Agent {
    network: FullyConnected,
}

impl Agent {
    fn new(dims: &[usize]) -> Self {
        assert!(!dims.is_empty());

        Agent {
            network: FullyConnected::network(dims),
        }
    }

    fn act(&self, state: &[f32]) -> usize {
        self.network.predict(state).argmax().unwrap()
    }

    fn child(a1: &Agent, a2: &Agent) -> Self {
        let mut rng = rand::thread_rng();
        let mut child_weights = Vec::with_capacity(a1.network.get_shape().len() - 1);
        let mut child_biases = Vec::with_capacity(a1.network.get_shape().len() - 1);
        let a1_weights = a1.network.get_weights();
        let a2_weights = a2.network.get_weights();
        let a1_biases = a1.network.get_biases();
        let a2_biases = a2.network.get_biases();
        let shape = a1.network.get_shape();
        for i in 0..shape.len() - 1 {
            let mut layer_weights = Array2::default((shape[i + 1], shape[i]));
            layer_weights.assign(a1_weights[i]);
            let mut layer_biases = Array1::default(shape[i + 1]);
            layer_biases.assign(a1_biases[i]);

            for j in 0..shape[i + 1] {
                if rng.gen_bool(0.5) {
                    layer_biases[j] = a2_biases[i][j];
                }
                for k in 0..shape[i] {
                    if rng.gen_bool(0.5) {
                        let weight = layer_weights.get_mut((j, k)).unwrap();
                        let a2_layer_weights = a2_weights[i];
                        *weight = *a2_layer_weights.get((j, k)).unwrap();
                    }
                }
            }
            child_weights.push(layer_weights);
            child_biases.push(layer_biases);
        }
        Self {
            network: FullyConnected::build_stack(&child_weights, &child_biases),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_child() {
        let a1 = Agent::new(&[2, 3, 1]);
        let a2 = Agent::new(&[2, 3, 1]);
        let child = Agent::child(&a1, &a2);

        assert_ne!(child.network, a1.network);
        assert_ne!(child.network, a2.network);
        assert_eq!(child.network.get_shape(), a1.network.get_shape());
    }

    #[test]
    fn test_agent_action() {
        let agent = Agent::new(&[2, 3, 1]);
        let state = [0., 1.];
        let actions = 2;
        let agent_action = agent.act(&state);

        assert!(agent_action >= 0);
        assert!(agent_action < actions);
    }
}
