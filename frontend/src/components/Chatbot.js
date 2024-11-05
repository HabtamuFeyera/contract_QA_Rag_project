// src/components/Chatbot.js
import React, { useState } from 'react';

const Chatbot = () => {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);

    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    const handleSendMessage = async () => {
        if (!input) return;

        // Add user message to the chat
        setMessages((prev) => [...prev, { sender: 'user', text: input }]);

        // Call your backend API
        try {
            const response = await fetch('http://localhost:8000/query/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: input }),  // Use 'question' to match your model
            });

            if (!response.ok) {
                const error = await response.json();
                console.error("Error response:", error);
                setMessages((prev) => [...prev, { sender: 'bot', text: "Sorry, I couldn't fetch a response." }]);
                return;
            }

            const data = await response.json();
            setMessages((prev) => [...prev, { sender: 'bot', text: data.answer }]);
        } catch (error) {
            console.error("Fetch error:", error);
            setMessages((prev) => [...prev, { sender: 'bot', text: "Sorry, I couldn't fetch a response." }]);
        }
        
        setInput('');
    };

    return (
        <div>
            <h1>Contract Advisor Chatbot</h1>
            <div>
                {messages.map((msg, index) => (
                    <div key={index} className={msg.sender}>
                        <strong>{msg.sender === 'user' ? 'You' : 'Bot'}: </strong>
                        {msg.text}
                    </div>
                ))}
            </div>
            <input type="text" value={input} onChange={handleInputChange} />
            <button onClick={handleSendMessage}>Send</button>
        </div>
    );
};

export default Chatbot;
