function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message) {
        addMessage(message, true);
        userInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            addMessage(data.response, false);
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Error processing request.', false);
        });
    }
}

function addMessage(message, isUser) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'agent-message'}`;

    if (!isUser) {
        typeMessage(messageDiv, message);
    } else {
        messageDiv.textContent = message;
    }

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function typeMessage(element, message, index = 0) {
    if (index < message.length) {
        element.textContent += message.charAt(index);
        setTimeout(() => typeMessage(element, message, index + 1), 15);
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

window.onload = function() {
    document.getElementById('userInput').focus();
    addMessage('Welcome! Try /analyze <coin> for market analysis or /help for commands.', false);
};
