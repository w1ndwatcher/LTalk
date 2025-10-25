import React, { useState, useRef } from 'react';
import Message from './Message';

function ChatBox({ chat, onSend, loading, setLoading }) {
  const [input, setInput] = useState("");
  const abortControllerRef = useRef(null);

  const suggestions = [
    "What is LearnTrail?",
    "What courses do you have?",
    "What is Artificial Intelligence?",
  ];

  const handleSend = async (prompt) => {
    const question = prompt || input.trim();
    if (!question) return;

    setInput("");
    setLoading(true);

    let botMsg = "";
    onSend(question, "");

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch("http://localhost:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: abortControllerRef.current.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        botMsg += decoder.decode(value);
        onSend(question, botMsg);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        botMsg += "\n🛑 Generation stopped by user.";
      } else {
        botMsg += "\n[ERROR]: " + err.message;
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
    }
  };

  return (
    <div className="d-flex flex-column flex-grow-1">
      <div className="flex-grow-1 p-3 d-flex flex-column" style={{ overflowY: "auto" }}>
        {/* Show chat messages if any */}
        {chat.length > 0 ? (
          chat.map((msg, idx) => (
            <React.Fragment key={idx}>
              <Message text={msg.userMsg} type="user" />
              <Message text={msg.botMsg} type="bot" />
            </React.Fragment>
          ))
        ) : (
          // Otherwise show prompt suggestions
          <div className="d-flex flex-column align-items-start justify-content-center h-100 text-center">
            <h5 className="mb-3">💡 Try asking me:</h5>
            <div className="d-flex flex-column gap-2 w-100">
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  className="btn btn-outline-primary text-start"
                  onClick={() => handleSend(s)}
                  disabled={loading}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="input-group p-3 border-top gap-2">
        <input
          type="text"
          className="form-control"
          placeholder="Ask me anything about LearnTrail..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={loading}
        />
        <button
          className="btn btn-warning"
          onClick={() => handleSend()}
          disabled={loading}
        >
          {loading ? "Thinking..." : "Send"}
        </button>
        <button
          className="btn btn-danger"
          onClick={handleStop}
          disabled={!loading}
        >
          Stop
        </button>
      </div>
    </div>
  );
}

export default ChatBox;