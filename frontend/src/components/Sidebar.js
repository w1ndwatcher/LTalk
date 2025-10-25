import React from 'react';

function Sidebar({ chats, onNewChat, onSelectChat }) {
  return (
    <div className="bg-dark text-white p-3 sidebar" style={{ width: '250px' }}>
      <h5 className="text-warning">🧠 LTalk</h5>
      <button className="btn btn-warning btn-sm my-2 w-100" onClick={onNewChat}>
        ➕ New Chat
      </button>

      <div className="chat-history mt-3">
        <h6 className="text-light">📜 Past Chats</h6>
        {chats.map((chat, idx) => (
          <div
            key={idx}
            className="small py-1 border-bottom border-light"
            style={{ cursor: "pointer" }}
            onClick={() => onSelectChat(idx)}
          >
            Chat #{idx + 1}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Sidebar;