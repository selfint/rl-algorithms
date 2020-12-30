use rl_algorithms::q_learning::QLearner;
use rl_environments::jump_environment::JumpEnvironment;

use std::{thread, time};

fn main() {
    // notice that state type is detected automatically, in line 18
    let mut learner = QLearner::new(2, 1., 0.1);

    for e in 0..10 {
        let mut env = JumpEnvironment::new(7);
        let mut score: i128 = 0;

        while !env.done {
            let env_state = env.raw_state();
            let action = learner.act(&env_state);
            if let 1 = action {
                env.jump();
            }

            let reward = env.update();
            score += reward as i128;

            let next_state = env.raw_state();
            learner.learn(&env_state, &next_state, action, reward as f32);

            // clear console and reset cursor
            print!("\x1B[2J\x1B[1;1H");

            println!("{}\nepoch={} score={}", &env, e, score);
            thread::sleep(time::Duration::from_millis(100));
        }
    }
}
