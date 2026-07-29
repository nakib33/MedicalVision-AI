/**
 * Brain Tumor MRI — Project-Specific Script
 *
 * This file can extend or override the shared main.js behaviour.
 * Currently it uses the shared implementation — add custom logic here if needed.
 */

// Brain tumor specific init (if any custom behaviour is needed)
document.addEventListener('DOMContentLoaded', () => {
    // Set page title accent
    const header = document.querySelector('h1');
    if (header) {
        header.style.borderLeft = '4px solid #805ad5';
        header.style.paddingLeft = '1rem';
    }
});
