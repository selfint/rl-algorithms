use rand::{thread_rng, Rng};
use std::{collections, hash::Hash};

pub struct QLearner<S>
where
    S: Eq + Hash,
{
    q_table: collections::HashMap<S, Vec<f32>>,
    actions: usize,
    epsilon: f32,
    epsilon_threshold: f32,
}

impl<S> QLearner<S>
where
    S: Eq + Hash,
{
    pub fn new(actions: usize) -> Self {
        Self {
            q_table: collections::HashMap::new(),
            actions,
            epsilon: 1.,
            epsilon_threshold: 0.1,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl_environments::jump_environment::JumpEnvironment;

    #[test]
    fn test_act() {
        let mut env = JumpEnvironment::new(10);
        let learner = QLearner::new(2);
        let env_state = env.simple_state().unwrap();
        let action = learner.act(&env_state);
        if let 1 = action {
            env.jump()
        }
        env.update();
    }
}
