let chatHistory = []; // Initialize history

function startDirectSession(mode) {
    const companyInput = document.getElementById('company_input');

    // Auto-fill for presentation if empty
    if (!companyInput.value.trim() && companyInput.dataset.default_val) {
        companyInput.value = companyInput.dataset.default_val;
    } else if (!companyInput.value.trim()) {
        companyInput.value = 'Amazon'; // Fallback FAANG dummy data
    }

    const company = companyInput.value.trim();
    if (!company) {
        alert('Please enter a target company.');
        return;
    }

    if (mode === 'voice') {
        // Redirect for Voice Mode
        const voiceUrl = "{{ voice_interview_url|escapejs }}";
        if (voiceUrl && voiceUrl !== '#' && voiceUrl !== 'None') {
            window.open(voiceUrl, '_blank');
        } else {
            alert("Voice service is currently configuring. Please try again later.");
        }
        return;
    }

    if (mode === 'text') {
        // Text Mode Logic
        initiateTextSession(company);
    }
}

function initiateTextSession(companyName) {
    const btn = document.getElementById('start-btn');
    if (btn) {
        btn.innerText = 'Initializing...';
        btn.style.opacity = '0.7';
    }

    const formData = new FormData();
    formData.append('company_name', companyName);
    formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);

    fetch('', {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                if (btn) {
                    btn.innerText = 'Start Text Session';
                    btn.style.opacity = '1';
                }
                return;
            }

            // Hide setup block and show chat
            document.getElementById('setup-view').style.display = 'none';
            const chatView = document.getElementById('chat-view');
            chatView.style.display = 'flex';

            document.getElementById('chat-session-title').innerText = 'Interview @ ' + companyName;

            // Add Initial Bot Message
            const messagesDiv = document.getElementById('chat-messages-container');
            const initialMsg = `Hello! I see you're preparing for ${companyName}. I've reviewed your resume and I'm ready to help you prepare. Let's start with a simple question: tell me about yourself?`;
            messagesDiv.innerHTML = `
                <div class="message ai-message">
                    <strong>AI Recruiter:</strong><br>
                    ${initialMsg}
                </div>
            `;
            // Add to history
            chatHistory = [{ role: 'model', content: initialMsg }];

            // Show recommendations if any
            if (data.recommendations && data.recommendations.length > 0) {
                const recArea = document.getElementById('recommendations-area');
                const recList = document.getElementById('recommendations-list');
                recList.innerHTML = '';

                data.recommendations.forEach(rec => {
                    const badge = document.createElement('span');
                    badge.className = 'tag';
                    badge.style.padding = '5px 12px';
                    badge.style.borderRadius = '15px';
                    badge.style.fontSize = '0.85rem';
                    badge.innerText = rec;
                    recList.appendChild(badge);
                });
                recArea.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Something went wrong. Please try again.');
            if (btn) {
                btn.innerText = 'Start Text Session';
                btn.style.opacity = '1';
            }
        });
}

// Unified Chat Logic matching Therapy Chat
const chatInput = document.getElementById('user-message');

chatInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        sendInterviewMessage();
    }
});

function sendInterviewMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add User Message locally
    const messagesDiv = document.getElementById('chat-messages-container');
    messagesDiv.innerHTML += `<div class="message user-message">${message}</div>`;
    chatInput.value = '';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Add to history
    chatHistory.push({ role: 'user', content: message });

    // Send to backend
    const formData = new FormData();
    formData.append('user_message', message);
    formData.append('history', JSON.stringify(chatHistory));
    formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);

    // Simulated Lag for realism
    const indicator = document.getElementById('typingIndicator');
    messagesDiv = document.getElementById('chat-messages-container');
    messagesDiv.appendChild(indicator); // Move indicator to bottom
    indicator.style.display = 'flex';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    setTimeout(() => {
        fetch('chat/', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                let reply = "I'm having trouble connecting right now.";
                if (data.reply) {
                    reply = data.reply;
                } else if (data.content && data.content.parts) {
                    reply = data.content.parts[0].text;
                } else if (data.content) {
                    reply = data.content;
                } else if (data.text) {
                    reply = data.text;
                }

                if (reply && typeof reply === 'object' && reply.parts) {
                    reply = reply.parts[0].text;
                }

                indicator.style.display = 'none';

                messagesDiv.innerHTML += `
                <div class="message ai-message">
                    <strong>AI Recruiter:</strong><br>
                    ${reply}
                </div>
                `;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;

                // Add AI response to history
                chatHistory.push({ role: 'model', content: reply });
            })
            .catch(err => {
                console.error(err);
                indicator.style.display = 'none';
                messagesDiv.innerHTML += `<div class="message system-message">Error connecting to AI.</div>`;
            });
    }, 100);
}
