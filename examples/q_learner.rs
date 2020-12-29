use rl_algorithms::q_learning::QLearner;
use rl_environments::jump_environment::JumpEnvironment;

use std::{thread, time};

fn main() {
    let mut learner = QLearner::new(2, 0.5, 0.1);
    for e in 0..1000 {
        let mut env = JumpEnvironment::new(10);
        let mut score: i128 = 0;

        while !env.done {
            if let Some(env_state) = env.simple_state() {
                let action = learner.act(&env_state);
                if let 1 = action {
                    env.jump();
                }

                let reward = env.update();
                score += reward as i128;

                if let Some(next_state) = env.simple_state() {
                    learner.learn(&env_state, &next_state, action, reward as f32);
                }
            } else {
                env.update();
            }

            if e > 950 || score > 10 {
                // clear console and reset cursor
                print!("\x1B[2J\x1B[1;1H");

                println!("{}\nscore={} epoch={}", &env, score, e);
                thread::sleep(time::Duration::from_millis(100));
            }
        }
        if e > 950 {
            println!("DEAD");
            thread::sleep(time::Duration::from_millis(500));
        }
    }
}
