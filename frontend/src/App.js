import React, { useState } from 'react';
import './App.css';  // Import custom styles
import ChatBox from './components/ChatBox';  // Main chat window
import Sidebar from './components/Sidebar';  // Sidebar component
import studentImage from './assets/student.jpg'; // images

function App() {
  const [chats, setChats] = useState([]); // array of { userMsg, botMsg }
  const [currentChat, setCurrentChat] = useState([]);   // messages in current open chat
  const [loading, setLoading] = useState(false);  // is bot thinking?

  // Add new message to current chat
  const addMessage = (userMsg, botMsg) => {
    const newMsg = { userMsg, botMsg };
    const updatedChat = [...currentChat, newMsg];
    setCurrentChat(updatedChat);
  };

  // Save current chat to history
  const saveChat = () => {
    setChats([...chats, currentChat]);
    setCurrentChat([]);
  };

  return (
    <div className="app-container d-flex">
      <Sidebar
        chats={chats}
        onNewChat={saveChat}
        onSelectChat={(index) => setCurrentChat(chats[index])}
      />
      <div className="chat-section flex-grow-1">
        <header className="chat-header p-3 text-white d-flex align-items-center justify-content-between">
          <h4 className="mb-0">💬 LearnTrail Chatbot</h4>
          <img src={studentImage} alt="Student" height="40" />
        </header>
        <ChatBox
          chat={currentChat}
          onSend={addMessage}
          loading={loading}
          setLoading={setLoading}
        />
      </div>
    </div>
  );
}

export default App;