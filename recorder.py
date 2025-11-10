from collections import deque

class Recorder:
    """Record last 5 conversations"""
    def __init__(self, memory_size: int = 5):
        # create history deque with fixed memoery size to avoid long context
        self._conversation_history = deque(maxlen=memory_size)

    @property
    def history(self):
        """call conversation history"""
        return  self._conversation_history
    
    def append_history(self, query: str, response: str):
        """Add an exchange to conversation history."""
        self._conversation_history.append({
            "query": query,
            "response": response,
            # "entities": self.extract_entities(query + " " + response)
        })

