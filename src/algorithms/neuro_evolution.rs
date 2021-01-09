use ndarray::prelude::*;
use ndarray_rand::rand::rngs::ThreadRng;
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::WeightedIndex;

use ndarray_stats::QuantileExt;

use ndarray_rand::rand::seq::IteratorRandom;
use neuron::layers::relu::ReLuLayer;
use neuron::network::{FeedForwardNetwork, FeedForwardNetworkTrait};
use neuron::Layer;

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

    pub fn crossover(&self, other: &Self) -> Self {
        let mut child_network = self.network.clone();
        self.crossover_weights(&mut child_network, other.network.get_weights());
        self.crossover_biases(&mut child_network, other.network.get_biases());

        Self {
            network: child_network,
        }
    }

    fn crossover_weights(
        &self,
        child_network: &mut FeedForwardNetwork<ReLuLayer>,
        other_weights: Vec<&Array2<f32>>,
    ) {
        let mut rng = thread_rng();
        for (child_layer_weights, other_layer_weights) in child_network
            .layers
            .iter_mut()
            .map(|l| l.get_weights_mut())
            .zip(other_weights)
        {
            for i in 0..child_layer_weights.nrows() {
                for j in 0..child_layer_weights.ncols() {
                    if rng.gen_bool(0.5) {
                        child_layer_weights[[i, j]] = other_layer_weights[[i, j]];
                    }
                }
            }
        }
    }

    fn crossover_biases(
        &self,
        child_network: &mut FeedForwardNetwork<ReLuLayer>,
        other_biases: Vec<&Array1<f32>>,
    ) {
        let mut rng = thread_rng();
        for (child_layer_biases, other_layer_biases) in child_network
            .layers
            .iter_mut()
            .map(|l| l.get_biases_mut())
            .zip(other_biases.iter())
        {
            for i in 0..child_layer_biases.len() {
                if rng.gen_bool(0.5) {
                    child_layer_biases[i] = other_layer_biases[i];
                }
            }
        }
    }

    pub fn mutate(&mut self) {
        let mut rng = thread_rng();
        if rng.gen_bool(0.5) {
            self.mutate_weight();
        } else {
            self.mutate_bias();
        }
    }

    fn mutate_weight(&mut self) {
        let mut rng = thread_rng();

        let layer = rng.gen_range(0..self.network.layers.len() - 1);
        let layer_weights = self.network.layers[layer].get_weights_mut();

        let node = rng.gen_range(0..layer_weights.nrows());
        let previous_node = rng.gen_range(0..layer_weights.ncols());

        layer_weights[[node, previous_node]] = rng.gen_range(-0.01..0.01);
    }

    fn mutate_bias(&mut self) {
        let mut rng = thread_rng();

        let layer = rng.gen_range(0..self.network.layers.len() - 1);
        let layer_biases = self.network.layers[layer].get_biases_mut();

        let node = rng.gen_range(0..layer_biases.len());

        layer_biases[node] = rng.gen_range(-0.01..0.01);
    }
}

pub struct NeuroEvolution {
    pub population: Vec<Agent>,
    rng: ThreadRng,
    mutation_rate: f64,
}

impl NeuroEvolution {
    pub fn new(agents: usize, network_dimensions: &[usize], mutation_rate: f64) -> Self {
        Self {
            population: (0..agents)
                .map(|_| Agent::new(network_dimensions))
                .collect(),
            rng: thread_rng(),
            mutation_rate,
        }
    }

    pub fn new_generation(&mut self, scores: &Array1<i32>) {
        let mut new_population: Vec<Agent> = vec![];
        let dist = self.get_score_distribution(scores);
        for _ in 0..self.population.len() {
            let a1_index = self.rng.sample(&dist);
            let a1 = &self.population[a1_index];

            // don't crossover agent with itself
            let a2_dist = self.get_score_distribution(
                &scores
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &s)| if i != a1_index { Some(s) } else { None })
                    .collect(),
            );
            let a2 = &self.population[self.rng.sample(&a2_dist)];
            let mut child = a1.crossover(&a2);
            if self.rng.gen_bool(self.mutation_rate) {
                child.mutate();
            }

            new_population.push(child);
        }

        self.population = new_population;
    }

    fn get_score_distribution(&self, scores: &Array1<i32>) -> WeightedIndex<i32> {
        let min_score = *scores.min().unwrap();
        let max_score = *scores.max().unwrap();
        let weights = match min_score == max_score {
            true => Array1::ones(scores.len()),
            false => scores - min_score,
        };
        WeightedIndex::new(&weights).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_new_generation() {
        let agent_amount = 5;
        let mut neuro_evolution = NeuroEvolution::new(agent_amount, &[2, 3, 1], 0.01);
        let first_generation = neuro_evolution.population.clone();
        assert_eq!(neuro_evolution.population.len(), agent_amount);

        // score population
        let distribution = Uniform::new(-10, 10);
        let scores = Array1::random(agent_amount, distribution);
        neuro_evolution.new_generation(&scores);

        assert_eq!(neuro_evolution.population.len(), first_generation.len());
        assert_ne!(neuro_evolution.population, first_generation);
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

        a.mutate();

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
        let mut neuro_evolution = NeuroEvolution::new(0, &[], 0.);
        let a1 = Agent::new(&[2, 3, 1]);
        let a2 = Agent::new(&[2, 3, 1]);
        let child = a1.crossover(&a2);

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
