use ndarray::prelude::*;

use rl_algorithms::neuro_evolution::NeuroEvolution;
use rl_environments::jump_environment::JumpEnvironment;

use std::{thread, time};

fn main() {
    // notice that state type is detected automatically, in line 16
    let agents = 50;
    let mut learner = NeuroEvolution::new(agents, &[10 * 10 * 3, 2], 0.05);

    for e in 0..1000 {
        let mut agent_scores: Array1<i32> = Array1::zeros((agents,));
        let mut envs = vec![JumpEnvironment::new(10); agents];
        let mut max_score = 0;
        while envs.iter().any(|e| !e.done) {
            for (agent, (env, score)) in envs
                .iter_mut()
                .zip(agent_scores.iter_mut())
                .enumerate()
                .filter(|(_, (e, _))| !e.done)
            {
                if *score > env.max_reward {
                    env.done = true;
                    continue;
                }

                let state = env.observe().iter().flatten().map(|&x| x as f32).collect();
                let action = learner.act(agent, &state).unwrap();
                let reward = env.step(action);
                *score += reward as i32;

                if *score > max_score {
                    max_score = *score;
                }

                if *score >= env.max_reward {
                    *score += 100;
                }

                if e % 10 == 0 && agent == 0 {
                    // clear console and reset cursor
                    print!("\x1B[2J\x1B[1;1H");

                    println!("{}\nepoch={} score={}", &env, e, score);
                    thread::sleep(time::Duration::from_millis(100));
                }
            }

            println!("max score={:?}", max_score);
        }

        learner.new_generation(&agent_scores);

        println!("epoch={} scores: {:?}", e, agent_scores);
    }
}
