struct QLearner {
    q_table: Vec<Vec<f32>>,
}

impl QLearner {
    fn new(states: usize, actions: usize) -> Self {
        Self {
            q_table: vec![vec![0.; actions]; states],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl_environments::jump_environment::JumpEnvironment;

    #[test]
    fn q_table_contains_all_state_action_pairs() {
        let env = JumpEnvironment::new(10);

        // states are all possible distances from the
        // nearest wall, and if it is a high or low wall
        let states = (env.size - env.player_col) * 2;

        // jump or don't jump
        let actions = 2;

        let learner = QLearner::new(states, actions);

        assert_eq!(learner.q_table.len(), states);
        for state_action_values in learner.q_table {
            assert_eq!(state_action_values.len(), actions);
        }
    }
}
