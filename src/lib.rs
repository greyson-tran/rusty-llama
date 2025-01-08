use ollama_rs::generation::options::GenerationOptions;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use tokio::runtime::Runtime;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall() {
        let wrapd = LargeLanguageModel::new("llama3.2:3b".to_string(), 0.5, 1.1, 16, 0.45);
        println!("{}", wrapd.query("Why is the sky blue? Give a simple paragraph synopsis".to_string()));
        println!("{}", wrapd.query("What did you just say?".to_string()));
    }
}

pub struct LargeLanguageModel {
    runtime: Runtime,
    api: Ollama,

    generation_options: GenerationOptions,

    model: String,
}

impl LargeLanguageModel {
    pub fn new(mod_select: String, temperature: f32, repeat_penalty: f32, top_k: u32, top_p: f32) -> Self {
        Self {
            runtime: Runtime::new().expect("Runtime creation has failed."),
            api: Ollama::default(),
            generation_options: GenerationOptions::default()
                .temperature(temperature)
                .repeat_penalty(repeat_penalty)
                .top_k(top_k)
                .top_p(top_p),
            model: mod_select
        }
    }

    pub fn query(&self, prompt: String) -> String {
        (&self.runtime.block_on(
            async {
                let query = GenerationRequest::new(self.model.clone(), prompt).options(self.generation_options.clone());
                self.api.generate(query).await.expect("Responce generation has failed.").response.clone()
            }
        )).to_string()
    }
}