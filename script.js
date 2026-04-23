document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        alert('⚠️ Please enter text for sentiment analysis');
        textInput.focus();
        return;
    }
    
    // Loading state
    const btn = this;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span>⏳ Analyzing...</span><span>→</span>';
    btn.disabled = true;
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Show result section
            const resultSection = document.getElementById('resultSection');
            resultSection.style.display = 'block';
            
            // Update UI
            document.getElementById('resultEmoji').textContent = result.emoji;
            
            const sentimentLabel = document.getElementById('sentimentLabel');
            sentimentLabel.textContent = result.sentiment;
            sentimentLabel.className = `sentiment-text ${result.sentiment.toLowerCase()}`;
            
            // Update confidence
            const confidencePercent = document.getElementById('confidencePercent');
            const confidenceBar = document.getElementById('confidenceBar');
            
            confidencePercent.textContent = `${result.confidence}%`;
            confidenceBar.style.width = `${result.confidence}%`;
            
            // Set confidence bar color
            if (result.sentiment === 'Positive') {
                confidenceBar.style.background = 'linear-gradient(90deg, #10b981, #059669)';
            } else if (result.sentiment === 'Negative') {
                confidenceBar.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
            } else {
                confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
            }
            
            // Display original text
            document.getElementById('originalTextDisplay').textContent = text;
            
            // Scroll to results
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            alert(result.error || 'Error analyzing text');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error connecting to server. Make sure Flask is running!');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

// Keyboard shortcut: Ctrl+Enter
document.getElementById('textInput').addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        document.getElementById('analyzeBtn').click();
    }
});

// Character counter
const textarea = document.getElementById('textInput');
const counter = document.createElement('div');
counter.style.css
