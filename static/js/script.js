const companyForm = document.getElementById('company-form');
const chatMessages = document.querySelector('.chat-messages');

if (companyForm) {
    companyForm.addEventListener('submit', async function (event) {
        event.preventDefault(); // Stop the page from reloading

        const companyName = event.target.elements.company_name.value;
        const csrfToken = event.target.elements.csrfmiddlewaretoken.value;

        // 1. Display a "loading" message
        addMessage('Thinking...', 'system');

        try {
            // 2. Send the company name to our Django view
            const response = await fetch('/interview/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': csrfToken
                },
                body: new URLSearchParams({
                    'company_name': companyName
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // 3. Remove the loading message and display the AI's response
            chatMessages.removeChild(chatMessages.lastChild); // Remove "Thinking..."
            addMessage(data.content.parts[0].text, 'ai');

            // 4. Hide the initial company form and show the chat input form (we will build this next)
            companyForm.style.display = 'none';
            chatForm.style.display = 'flex';
            // We will add the next form for the chat conversation here in a future step.

        } catch (error) {
            chatMessages.removeChild(chatMessages.lastChild); // Remove "Thinking..."
            addMessage('Sorry, an error occurred. Please try again.', 'system');
            console.error('Fetch error:', error);
        }
    });
}

// A helper function to add messages to the chat window
function addMessage(text, sender) {
    if (!chatMessages) return;
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.innerText = text;
    chatMessages.appendChild(messageElement);
    // Scroll to the bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// static/script.js

// --- Add this new code to the bottom of your existing script ---

const chatForm = document.getElementById('chat-form');

// This event listener handles the ongoing conversation
if (chatForm) {
    chatForm.addEventListener('submit', async function (event) {
        event.preventDefault();

        const userMessage = event.target.elements.user_message.value;
        const csrfToken = event.target.elements.csrfmiddlewaretoken.value;

        // Display the user's own message immediately
        addMessage(userMessage, 'user');
        event.target.elements.user_message.value = ''; // Clear the input field

        addMessage('Thinking...', 'system');

        try {
            const response = await fetch('/interview/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': csrfToken
                },
                body: new URLSearchParams({
                    'user_message': userMessage
                    // We also need to send the conversation history
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            chatMessages.removeChild(chatMessages.lastChild); // Remove "Thinking..."
            addMessage(data.content.parts[0].text, 'ai');

        } catch (error) {
            chatMessages.removeChild(chatMessages.lastChild); // Remove "Thinking..."
            addMessage('Sorry, an error occurred. Please try again.', 'system');
            console.error('Fetch error:', error);
        }
    });
}

function toggleDropdown() {
    const dropdown = document.getElementById("profileDropdown");
    dropdown.classList.toggle("show");
}

// Close the dropdown if the user clicks outside of it
window.onclick = function (event) {
    if (!event.target.matches('.profile-avatar')) {
        var dropdowns = document.getElementsByClassName("dropdown-menu");
        var i;
        for (i = 0; i < dropdowns.length; i++) {
            var openDropdown = dropdowns[i];
            if (openDropdown.classList.contains('show')) {
                openDropdown.classList.remove('show');
            }
        }
    }
}
