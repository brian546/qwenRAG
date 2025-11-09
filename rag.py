from typing import List, Dict, Any
from .datatype import Document

class BasicRAG:
    def __init__(self, model, tokenizer, vector_store):
        self.model = model
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.prompt_instruction = """You are a helpful assistant. Reply briefly and only with information explicitly stated in the context that directly answers the question. If you cannot find an answer based on the provided information, say "I don't have enough information to answer that.". Use the following examples as a guide for your response style but not context:

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

        Now, use this context to answer the query:"""
        
    def _build_prompt(self, query: str, contexts: List[Document]) -> str:
        """Build a prompt combining the query and retrieved contexts."""
        context_str = "\n\n".join([doc.text for doc in contexts])
        
        prompt = f"""{self.prompt_instruction}
        Context:
        {context_str}

        Query: {query}

        Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, max_new_tokens: int = 200, device: str = 'cpu') -> str:
        """Generate a response using RAG."""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(query)
        
        # Build prompt
        prompt = self._build_prompt(query, relevant_docs)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.3, # less creative answer
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.split("Answer:")[-1].strip()
    
class MultiTurnRAG(BasicRAG):
    def __init__(self, model, tokenizer, vector_store):
        super().__init__(model, tokenizer, vector_store)
        self.query_processor = QueryProcessor()
        
    def _build_prompt(self, query: str, contexts: List[Document], include_history: bool = True) -> str:
        """Build a prompt with conversation history."""
        context_str = "\n\n".join([doc.text for doc in contexts])
        
        history_str = ""
        if include_history and self.query_processor.conversation_history:
            history = self.query_processor.conversation_history[-3:]  # Last 3 exchanges
            history_str = "\n".join([
                f"Query: {h['query']}\nAnswer: {h['response']}"
                for h in history
            ])
            history_str = f"\nPrevious conversation:\n{history_str}\n"
            
        prompt = f"""{self.prompt_instruction}
        Context:
        {context_str}
        {history_str}

        Query: {query}

        Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, max_new_tokens: int = 200) -> str:
        """Generate a response with conversation context."""
        # Process and expand query
        processed_query = self.query_processor.preprocess_query(query)
        expanded_query = self.query_processor.expand_query(processed_query)
        
        # Get response using the expanded query
        response = super().generate_response(expanded_query)
        
        # Update conversation history
        self.query_processor.add_to_history(query, response)
        
        return response