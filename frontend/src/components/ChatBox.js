import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";

function ChatBox({ chat, sessionId, onStart, onUpdate, loading, setLoading }) {
  const [input, setInput] = useState("");
  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to the latest message on every chat update — this fires on
  // every streamed token too, since `chat` changes each time onUpdate runs.
  // "auto" (instant) rather than "smooth" deliberately — smooth-scroll
  // animations queued dozens of times a second during streaming fight each
  // other and look janky; instant scroll just tracks the bottom cleanly.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [chat]);

  const suggestions = [
    "What is LearnTrail?",
    "What courses do you have?",
    "What is Artificial Intelligence?",
  ];

  const handleSend = async (prompt) => {
    const question = prompt || input.trim();
    if (!question) return;
    // Frontend-side nicety only — the backend enforces this authoritatively
    // and is what actually protects the app; never trust client-side checks alone.
    if (question.length > 1000) {
      onStart(question);
      onUpdate("That question is too long — please keep it under 1000 characters.", null);
      return;
    }

    setInput("");
    setLoading(true);
    onStart(question);

    let botMsg = "";
    let runId = null;
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: sessionId }),
        signal: abortControllerRef.current.signal,
      });

      // Backend attaches the run_id as a header so we can attribute
      // feedback (thumbs up/down) to the exact generation later.
      runId = response.headers.get("X-Run-Id");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        botMsg += decoder.decode(value);
        onUpdate(botMsg, runId);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        botMsg += "\n[Stopped by user]";
      } else {
        botMsg += "\n[ERROR]: " + err.message;
      }
      onUpdate(botMsg, runId);
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

  const handleFeedback = async (runId, score) => {
    if (!runId) return;
    try {
      await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId, score }),
      });
    } catch (err) {
      console.error("Feedback submission failed:", err);
    }
  };

  return (
    <div className="d-flex flex-column flex-grow-1" style={{ minHeight: 0 }}>
      <div className="flex-grow-1 p-3 d-flex flex-column" style={{ overflowY: "auto", minHeight: 0 }}>
        {/* Show chat messages if any */}
        {chat.length > 0 ? (
          chat.map((msg, idx) => (
            <React.Fragment key={idx}>
              <Message text={msg.userMsg} type="user" />
              <Message
                text={msg.botMsg}
                type="bot"
                runId={msg.runId}
                onFeedback={handleFeedback}
              />
            </React.Fragment>
          ))
        ) : (
          // Otherwise show prompt suggestions
          <div className="d-flex flex-column align-items-start justify-content-center h-100 text-center">
            <h5 className="mb-3">
              <i className="fa-solid fa-lightbulb me-2"></i>
              Try asking me:
            </h5>
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
        {/* Scroll target — kept at the very end of the message list */}
        <div ref={messagesEndRef} />
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