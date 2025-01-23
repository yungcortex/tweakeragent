function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message) {
        addMessage(`You: ${message}`, true);
        userInput.value = '';

        // Check if message is about crypto
        const cryptoKeywords = ['price', 'analyze', 'check', 'how is', 'how much', 'what is'];
        const commonTickers = ['btc', 'eth', 'sol', 'bnb', 'xrp', 'doge', 'ada', 'dot', 'matic', 'link'];

        const lowerMessage = message.toLowerCase();
        const isAskingAboutCrypto = (
            cryptoKeywords.some(keyword => lowerMessage.includes(keyword)) ||
            commonTickers.some(ticker => lowerMessage.includes(ticker)) ||
            /0x[a-fA-F0-9]{40}/.test(message)
        );

        let token = '';
        if (isAskingAboutCrypto) {
            token = commonTickers.find(ticker => lowerMessage.includes(ticker)) || '';
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
    scrollToBottom();
}

// Faster typing speed (5ms instead of 15ms)
function typeMessage(element, message, index = 0) {
    if (index < message.length) {
        element.textContent += message.charAt(index);
        setTimeout(() => {
            typeMessage(element, message, index + 1);
            scrollToBottom();  // Scroll while typing
        }, 5);  // Reduced from 15ms to 5ms for faster typing
    }
}

// New function to handle scrolling
function scrollToBottom() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
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
