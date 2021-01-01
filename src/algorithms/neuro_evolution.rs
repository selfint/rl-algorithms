struct Agent;

impl Agent {
    fn new() -> Self {
        Agent
    }

    fn act<S>(&self, state: &S) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_action() {
        let agent = Agent::new();
        let state = [0, 1];
        let actions = 2;
        let agent_action = agent.act(&state);

        assert!(agent_action >= 0);
        assert!(agent_action < actions);
    }
}
