// Handle message sending
function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message) {
        // Add user message to chat
        addMessage(`You: ${message}`, true);

        // Clear input field
        userInput.value = '';

        // Send message to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Add AI response to chat with "Agent: " prefix if it doesn't already have it
            const response = data.response.startsWith('Agent:') ?
                data.response :
                `Agent: ${data.response}`;
            addMessage(response, false);
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Agent: Error processing command. Please try again.', false);
        });
    }
}

// Add message to chat window
function addMessage(message, isUser) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'agent-message'}`;

    // Add typing effect for agent messages
    if (!isUser) {
        typeMessage(messageDiv, message);
    } else {
        messageDiv.textContent = message;
    }

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Typing effect for agent messages
function typeMessage(element, message, index = 0) {
    if (index < message.length) {
        element.textContent += message.charAt(index);
        setTimeout(() => typeMessage(element, message, index + 1), 15);
    }
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Auto-focus input field on page load
window.onload = function() {
    document.getElementById('userInput').focus();

    // Add initial message
    addMessage('Agent: initializing cynicism protocols... try /analyze <coin> or /help', false);
};
