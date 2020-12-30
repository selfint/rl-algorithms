use rand::{thread_rng, Rng};
use std::{collections, hash::Hash};

pub struct QLearner<S>
where
    S: Eq + Hash + Clone,
{
    q_table: collections::HashMap<S, Vec<f32>>,
    actions: usize,
    epsilon: f32,
    epsilon_threshold: f32,
    lr: f32,
    gamma: f32,
}

impl<S> QLearner<S>
where
    S: Eq + Hash + Clone,
{
    pub fn new(actions: usize, lr: f32, gamma: f32) -> Self {
        Self {
            q_table: collections::HashMap::new(),
            actions,
            epsilon: 1.,
            epsilon_threshold: 0.1,
            lr,
            gamma,
        }
    }

    pub fn act(&self, state: &S) -> usize {
        if self.epsilon <= self.epsilon_threshold {
            if let Some(action_values) = self.q_table.get(state) {
                let max_action_value: &f32 = action_values
                    .iter()
                    .max_by(|&x, &y| {
                        x.partial_cmp(y)
                            .expect("Action values contained NaN - can't find max")
                    })
                    .expect("Failed to find max action value");
                let best_action = action_values
                    .iter()
                    .position(|v: &f32| (v - max_action_value).abs() < f32::EPSILON)
                    .unwrap();
                return best_action;
            } else {
                return thread_rng().gen_range(0..self.actions);
            }
        }

        thread_rng().gen_range(0..self.actions)
    }

    pub fn learn(&mut self, state: &S, next_state: &S, action: usize, reward: f32) {
        let next_state_max_value = match self.q_table.get(next_state) {
            Some(action_values) => action_values
                .iter()
                .max_by(|&x, &y| {
                    x.partial_cmp(y)
                        .expect("Action values contained NaN - can't find max")
                })
                .expect("Failed to find max action value"),
            None => {
                self.q_table
                    .insert(next_state.clone(), vec![0.; self.actions]);
                &0.
            }
        };

        let update = self.lr * (reward + self.gamma * next_state_max_value);
        match self.q_table.get(state) {
            Some(action_values) => {
                let action_value = action_values[action];
                self.q_table.get_mut(state).unwrap()[action] =
                    (1.0 - self.lr) * action_value + update;
            }
            None => {
                let action_value = reward;
                let new_action_value: f32 = (1.0 - self.lr) * action_value + update;
                let mut new_action_values = vec![0.; self.actions];
                new_action_values[action] = new_action_value;
                self.q_table.insert(state.clone(), new_action_values);
            }
        };

        self.epsilon *= 0.9;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_act() {
        let learner = QLearner::new(2, 0.1, 0.1);
        let env_state = vec![1, 0];
        let action = learner.act(&env_state);
        assert!(action < 2);
    }

    #[test]
    fn test_learn() {
        let mut learner = QLearner::new(2, 0.1, 0.1);
        let env_state = vec![0, 1];
        let action = learner.act(&env_state);
        let reward = 1;
        let next_state = vec![1, 0];
        learner.learn(&env_state, &next_state, action, reward as f32);
    }
}
