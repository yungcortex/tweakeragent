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
        // Split the message on our delimiter and create spans for each line
        const lines = message.split('|||');
        if (lines.length > 1) {
            // If it's a chart message (has multiple lines)
            messageDiv.style.whiteSpace = 'pre';
            typeMessageWithLines(messageDiv, lines);
        } else {
            // Regular message
            typeMessage(messageDiv, message);
        }
    } else {
        messageDiv.textContent = message;
    }

    messagesDiv.appendChild(messageDiv);
    scrollToBottom();
}

// New function to handle typing multiple lines
function typeMessageWithLines(element, lines, lineIndex = 0, charIndex = 0) {
    if (lineIndex < lines.length) {
        if (charIndex === 0 && lineIndex > 0) {
            // Add a line break before each new line except the first
            element.innerHTML += '\n';
        }
        
        if (charIndex < lines[lineIndex].length) {
            element.innerHTML += lines[lineIndex].charAt(charIndex);
            setTimeout(() => {
                typeMessageWithLines(element, lines, lineIndex, charIndex + 1);
                scrollToBottom();
            }, 5);
        } else {
            // Move to next line
            setTimeout(() => {
                typeMessageWithLines(element, lines, lineIndex + 1, 0);
                scrollToBottom();
            }, 5);
        }
    }
}

// Original typeMessage function (keep this for regular messages)
function typeMessage(element, message, index = 0) {
    if (index < message.length) {
        element.textContent += message.charAt(index);
        setTimeout(() => {
            typeMessage(element, message, index + 1);
            scrollToBottom();
        }, 5);
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
