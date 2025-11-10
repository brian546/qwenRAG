from typing import List
from document import Document
from recorder import Recorder

class BasicRAG:
    """Basic RAG system based on context retrived with functions that build prompt and generate response"""
    def __init__(self, model, tokenizer, vector_store, device='cuda'):
        self.device = device
        self.model = model
        self.model.to(device)
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.assistant_instruction = """You are a helpful assistant. Reply briefly and only with information explicitly stated in the context that directly answers the question. If you cannot find an answer based on the provided information, say "I don't have enough information to answer that.". Use the following examples as a guide for your response style:

        Example 1:
        Context: The capital of Germany is the city of Berlin. The capital of France is Paris.
        Query: What is the capital of Germany?
        Answer: The capital of Germany is Berlin.

        Example 2:
        Context: William Shakespeare was an English playwright. He wrote Romeo and Juliet in the 1590s.
        Query: Who wrote Romeo and Juliet?
        Answer: Romeo and Juliet was written by William Shakespeare.

        Example 3:
        Context: BeiJing is in China.
        Query: Where is New York?
        Answer: I don't have enough information to answer that.

        """
        
    def _build_prompt(self, query: str, contexts: List[Document]) -> str:
        """Build a prompt combining the query and retrieved contexts."""
        context_str = "\n\n".join([doc.text for doc in contexts])
        
        prompt = f"""{self.assistant_instruction}
        
        Now, use this context to answer the query:
        Context:
        {context_str}

        Query: {query}

        Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, max_new_tokens: int = 200) -> str:
        """Generate a response using RAG."""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(query)
        
        # Build prompt
        prompt = self._build_prompt(query, relevant_docs)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.3, # less creative answer
            do_sample=True,            # optional: makes it less repetitive
            pad_token_id=self.tokenizer.eos_token_id 
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.split("Answer:")[-1].strip()
    
class MultiTurnRAG(BasicRAG):
    def __init__(self, model, tokenizer, vector_store, memory_size = 5):
        super().__init__(model, tokenizer, vector_store)
        self.recorder = Recorder(memory_size=memory_size)
        
    def _build_prompt(self, query: str, contexts: List[Document], include_history: bool = True) -> str:
        """Build a prompt with conversation history."""
        context_str = "\n\n".join([doc.text for doc in contexts])
        
        history_str = ""
        if include_history and self.recorder.history:
            # print(history)
            history = self.recorder.history
            history_str = "\n".join([
                f"Query: {h['query']}\nAnswer: {h['response']}"
                for h in history 
            ])
            history_str = f"\nPrevious conversation:\n{history_str}\n"

        # integrate conversation history and context into prompt
        prompt = f"""{self.assistant_instruction}

        {history_str}

        Now, use this context to answer the query:
        Context:
        {context_str}

        Query: {query}

        Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, max_new_tokens: int = 200) -> str:
        """Generate a response with conversation context. And update history"""

        response = super().generate_response(query, max_new_tokens)
        
        # Update conversation history
        self.recorder.append_history(query, response)
        
        return response
