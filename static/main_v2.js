function addMessage(message, isUser) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = isUser ? `You: ${message}` : message;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    if (message) {
        addMessage(message, true);
        input.value = '';

        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            addMessage(data.response, false);
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Error getting response. Please try again.', false);
        });
    }
}

// Focus input on page load
window.onload = function() {
    document.getElementById('userInput').focus();
};
