function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message) {
        addMessage(`You: ${message}`, true);
        userInput.value = '';

        // Determine which endpoint to use based on the command
        let endpoint = '/chat';
        if (message.startsWith('/analyze')) {
            endpoint = '/ask';  // Use the /ask endpoint for analysis commands
        }

        fetch(endpoint, {
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
            if (data.response) {
                addMessage(data.response, false);
            } else {
                addMessage('Error: No response from server', false);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Error processing request. Please try again.', false);
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

function analyzeToken(symbol) {
    const message = `/analyze ${symbol}`;
    document.getElementById('userInput').value = message;
    sendMessage();
}

window.onload = function() {
    document.getElementById('userInput').focus();
    addMessage('Welcome! Try /analyze <coin> for market analysis or /help for commands.', false);
};
