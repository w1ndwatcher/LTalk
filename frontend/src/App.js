import React, { useState, useRef } from 'react';
import './App.css';  // Import custom styles
import ChatBox from './components/ChatBox';  // Main chat window
import Sidebar from './components/Sidebar';  // Sidebar component

function generateSessionId() {
  if (window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID();
  }
  // Fallback for older browsers
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function App() {
  const [chats, setChats] = useState([]); // array of { sessionId, messages: [{userMsg, botMsg, runId}] }
  const [currentSessionId, setCurrentSessionId] = useState(generateSessionId());
  const [currentChat, setCurrentChat] = useState([]);   // messages in current open chat
  const [loading, setLoading] = useState(false);  // is bot thinking?

  // Tracks which row in currentChat the in-flight response belongs to,
  // so streamed tokens update ONE row instead of appending a new one each time.
  const activeIndexRef = useRef(null);

  // Called once, right when a question is sent — creates the placeholder row
  const startMessage = (userMsg) => {
    setCurrentChat((prev) => {
      const updated = [...prev, { userMsg, botMsg: "", runId: null }];
      activeIndexRef.current = updated.length - 1;
      return updated;
    });
  };

  // Called on every streamed chunk — updates the same row in place
  const updateMessage = (botMsg, runId) => {
    setCurrentChat((prev) => {
      const idx = activeIndexRef.current;
      if (idx === null || !prev[idx]) return prev;
      const updated = [...prev];
      updated[idx] = {
        ...updated[idx],
        botMsg,
        runId: runId ?? updated[idx].runId,
      };
      return updated;
    });
  };

  // Save current chat to history, start a fresh session for the next one
  const saveChat = () => {
    if (currentChat.length > 0) {
      setChats((prev) => [...prev, { sessionId: currentSessionId, messages: currentChat }]);
    }
    setCurrentChat([]);
    setCurrentSessionId(generateSessionId());
    activeIndexRef.current = null;
  };

  // Reopen a past chat — restore its session_id too, so follow-up questions
  // in that chat still have the right history on the backend
  const selectChat = (index) => {
    const chat = chats[index];
    setCurrentChat(chat.messages);
    setCurrentSessionId(chat.sessionId);
    activeIndexRef.current = null;
  };

  return (
    <div className="app-container d-flex">
      <Sidebar
        chats={chats}
        onNewChat={saveChat}
        onSelectChat={selectChat}
      />
      <div className="chat-section flex-grow-1">
        <header className="chat-header p-3 text-white d-flex align-items-center justify-content-between">
          <h4 className="mb-0">
            <i className="fa-solid fa-comments me-2"></i>
            LearnTrail Chatbot
          </h4>
          {/* Files in public/ are served from the root path — no import
              needed, unlike files under src/assets/. */}
          <img src="/ltlogo.jpg" alt="LearnTrail" height="40" />
        </header>
        <ChatBox
          chat={currentChat}
          sessionId={currentSessionId}
          onStart={startMessage}
          onUpdate={updateMessage}
          loading={loading}
          setLoading={setLoading}
        />
      </div>
    </div>
  );
}

export default App;