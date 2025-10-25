import React from 'react';

function Message({ text, type }) {
  return (
    <div
      className={`message ${type === "user" ? "user-msg ms-auto" : "bot-msg me-auto"}`}
    >
      {text}
    </div>
  );
}

export default Message;