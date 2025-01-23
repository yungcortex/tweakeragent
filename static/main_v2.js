function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message) {
        addMessage(`You: ${message}`, true);
        userInput.value = '';

        // Check if the message is asking about a crypto price or analysis
        const cryptoKeywords = ['price', 'analyze', 'check', 'how much', 'how is', 'what is'];
        const commonTickers = ['btc', 'eth', 'sol', 'bnb', 'xrp', 'doge', 'ada', 'dot', 'matic', 'link'];

        // Convert message to lowercase for easier matching
        const lowerMessage = message.toLowerCase();

        // Check if message contains a crypto keyword and either a common ticker or looks like a contract address
        const isAskingAboutCrypto = (
            cryptoKeywords.some(keyword => lowerMessage.includes(keyword)) ||
            commonTickers.some(ticker => lowerMessage.includes(ticker)) ||
            /0x[a-fA-F0-9]{40}/.test(message) // Check for ETH-style addresses
        );

        // Extract the potential token identifier
        let token = '';
        if (isAskingAboutCrypto) {
            // Try to find a common ticker in the message
            token = commonTickers.find(ticker => lowerMessage.includes(ticker)) || '';

            // If no common ticker found, check for contract address
            if (!token) {
                const addressMatch = message.match(/0x[a-fA-F0-9]{40}/);
                if (addressMatch) {
                    token = addressMatch[0];
                }
            }
        }

        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: isAskingAboutCrypto ? `/analyze ${token || message}` : message
            })
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

window.onload = function() {
    document.getElementById('userInput').focus();
    addMessage('Hi! I can help you check crypto prices and market analysis. Try asking about any coin like "How is BTC doing?" or "Check SOL price"', false);
};
