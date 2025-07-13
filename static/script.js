document.addEventListener('DOMContentLoaded', function() {
    // Animate elements as they scroll into view
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.prediction-card, .confidence-breakdown');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (elementPosition < screenPosition) {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }
        });
    };
    
    // Set initial state for animation
    const animatedElements = document.querySelectorAll('.prediction-card, .confidence-breakdown');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'all 0.6s ease-out';
    });
    
    // Trigger on load and scroll
    animateOnScroll();
    window.addEventListener('scroll', animateOnScroll);
    
    // Add copy functionality
    const copyButton = document.createElement('button');
    copyButton.innerHTML = '<i class="fas fa-copy"></i> Copy Results';
    copyButton.classList.add('copy-btn');
    copyButton.addEventListener('click', () => {
        const resultsText = `Fake News Detection Result:\n\n` +
            `Prediction: ${document.querySelector('.prediction-card h3 span').textContent}\n` +
            `Confidence: ${document.querySelector('.confidence-meter span').textContent}\n\n` +
            `Full Breakdown:\n${Array.from(document.querySelectorAll('tbody tr'))
                .map(row => `${row.cells[0].textContent}: ${row.cells[1].textContent}`)
                .join('\n')}`;
        
        navigator.clipboard.writeText(resultsText)
            .then(() => {
                const originalText = copyButton.innerHTML;
                copyButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    copyButton.innerHTML = originalText;
                }, 2000);
            });
    });
    
    const shareSection = document.querySelector('.share-section');
    if (shareSection) {
        shareSection.appendChild(copyButton);
    }
});