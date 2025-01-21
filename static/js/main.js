const chatLog = document.getElementById('chat-log');
const userInput = document.getElementById('user-input');

userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function analyzeToken(token) {
    userInput.value = `/analyze ${token}`;
    sendMessage();
}

function random(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing';
    typingDiv.textContent = 'Agent is analyzing';
    chatLog.appendChild(typingDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
    return typingDiv;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage('You', message, 'user');
    userInput.value = '';

    const typingIndicator = showTypingIndicator();

    try {
        await new Promise(resolve => setTimeout(resolve, random(500, 1500)));
        
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        
        typingIndicator.remove();
        addMessage('Agent', data.response, 'agent');
    } catch (error) {
        typingIndicator.remove();
        addMessage('Agent', 'error processing your request... fitting.', 'agent');
    }
}

function addMessage(sender, message, className) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    messageDiv.textContent = `${sender}: ${message}`;
    chatLog.appendChild(messageDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// Initialize with welcome message
window.onload = function() {
    addMessage('Agent', 'initializing cynicism protocols... try /analyze <coin> or /help', 'agent');
};