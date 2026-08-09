import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function Message({ text, type, runId, onFeedback }) {
  const [given, setGiven] = useState(null); // 'up' | 'down' | null

  const handleFeedback = (score, label) => {
    if (given || !runId) return;
    setGiven(label);
    onFeedback(runId, score);
  };

  return (
    <div
      className={`message ${type === "user" ? "user-msg ms-auto" : "bot-msg me-auto"}`}
    >
      {/* User messages stay plain text — no need to interpret markdown in
          what the person typed. Bot messages render as markdown so
          **bold**, bullet lists, etc. actually display instead of showing
          raw asterisks/dashes. react-markdown does not execute embedded
          HTML by default, so this is safe even if a response somehow
          contained markup. */}
      {type === "bot" ? (
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        </div>
      ) : (
        <div>{text}</div>
      )}

      {type === "bot" && runId && text && (
        <div className="d-flex gap-2 mt-1">
          <button
            className={`btn btn-sm ${given === "up" ? "btn-success" : "btn-outline-secondary"}`}
            onClick={() => handleFeedback(1, "up")}
            disabled={!!given}
            title="Good response"
          >
            <i className="fa-solid fa-thumbs-up"></i>
          </button>
          <button
            className={`btn btn-sm ${given === "down" ? "btn-danger" : "btn-outline-secondary"}`}
            onClick={() => handleFeedback(0, "down")}
            disabled={!!given}
            title="Not helpful"
          >
            <i className="fa-solid fa-thumbs-down"></i>
          </button>
        </div>
      )}
    </div>
  );
}

export default Message;