use ollama_rs::generation::options::GenerationOptions;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use tokio::runtime::Runtime;

use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use ollama_rs::history::ChatHistory;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall() {
        let mut wrapd = LargeLanguageModel::new("deepseek-r1:1.5b".to_string(), 0.5, 1.1, 16, 0.45);
        println!("{}", wrapd.query_with_memory("Hola, Mundo!".to_string()));
        println!("{}", wrapd.query_with_memory("Why were you speaking in Spanish?".to_string()));
            // wrapd.query_with_memory(
            //     "Why is the sky blue? Give a simple paragraph synopsis".to_string()
            // )
    }
}

pub struct LargeLanguageModel {
    runtime: Runtime,
    api: Ollama,

    generation_options: GenerationOptions,

    model: String,

    history: Vec<ChatMessage>,
}

impl LargeLanguageModel {
    pub fn new(
        model_select: String,
        temperature: f32,
        repeat_penalty: f32,
        top_k: u32,
        top_p: f32,
    ) -> Self {
        Self {
            runtime: Runtime::new().expect("Runtime creation has failed."),
            api: Ollama::default(),
            generation_options: GenerationOptions::default()
                .temperature(temperature)
                .repeat_penalty(repeat_penalty)
                .top_k(top_k)
                .top_p(top_p),
            model: model_select,
            history: vec![],
        }
    }

    pub fn query(&self, prompt: String) -> String {
        (&self.runtime.block_on(async {
            let query = GenerationRequest::new(self.model.clone(), prompt)
                .options(self.generation_options.clone());
            self.api
                .generate(query)
                .await
                .expect("Responce generation has failed.")
                .response
                .clone()
        }))
            .to_string()
    }

    pub fn query_with_memory(&mut self, prompt: String) -> String {
        self.runtime.block_on(async {
            let result = self.api.clone()
                .send_chat_messages_with_history(
                    &mut self.history,
                    ChatMessageRequest::new(
                        self.model.clone(),
                        vec![ChatMessage::user(prompt)],
                    ),
                )
                .await;
            
            match result {
                Ok(answer) => answer.message.content,
                Err(e) => format!("Error: {:?}", e),
            }
        })
    }
}
