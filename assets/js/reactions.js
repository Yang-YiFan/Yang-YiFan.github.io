/**
 * Blog Post Emoji Reactions System
 * Provides Slack-style emoji reactions for blog posts
 * Uses localStorage for persistence across sessions
 */

class BlogReactions {
  constructor() {
    this.reactions = ['👍', '❤️', '😄', '🎉', '🤔', '👏', '🔥', '💡'];
    this.storageKey = 'blog-reactions';
    this.currentPage = this.getCurrentPageId();
    this.init();
  }

  getCurrentPageId() {
    // Use the current page path as a unique identifier
    return window.location.pathname;
  }

  init() {
    // Only initialize if we're on a blog post page
    if (this.isBlogPost()) {
      this.createReactionContainer();
      this.loadReactions();
      this.bindEvents();
    }
  }

  isBlogPost() {
    // Check if we're on a blog post page
    const path = window.location.pathname;
    
    // For GitHub Pages, blog posts can be:
    // 1. /blogs/something/something.md or .html
    // 2. /blogs/something/ (directory with index)
    // 3. Test page for development
    return (path.includes('/blogs/') && (path.includes('.md') || path.includes('.html'))) ||
           (path.includes('/blogs/') && document.querySelector('h1, h2, h3')) ||
           path.includes('test-reactions.html'); // For testing
  }

  createReactionContainer() {
    const container = document.createElement('div');
    container.className = 'blog-reactions';
    container.innerHTML = `
      <div class="reactions-header">
        <h4>👋 What did you think of this post?</h4>
      </div>
      <div class="reactions-buttons">
        ${this.reactions.map(emoji => `
          <button class="reaction-btn" data-emoji="${emoji}">
            <span class="emoji">${emoji}</span>
            <span class="count">0</span>
          </button>
        `).join('')}
      </div>
      <div class="reactions-footer">
        <small>Reactions are stored locally in your browser</small>
      </div>
    `;

    // Insert before the footer
    const footer = document.querySelector('footer');
    if (footer) {
      footer.parentNode.insertBefore(container, footer);
    } else {
      // Fallback: append to the section element
      const section = document.querySelector('section');
      if (section) {
        section.appendChild(container);
      }
    }
  }

  bindEvents() {
    const buttons = document.querySelectorAll('.reaction-btn');
    buttons.forEach(button => {
      button.addEventListener('click', (e) => {
        e.preventDefault();
        const emoji = button.dataset.emoji;
        this.toggleReaction(emoji, button);
      });
    });
  }

  toggleReaction(emoji, button) {
    const data = this.getStoredData();
    const pageData = data[this.currentPage] || {};
    const userReactions = pageData.userReactions || [];
    const counts = pageData.counts || {};

    // Initialize count if it doesn't exist
    if (!counts[emoji]) {
      counts[emoji] = 0;
    }

    if (userReactions.includes(emoji)) {
      // Remove reaction
      const index = userReactions.indexOf(emoji);
      userReactions.splice(index, 1);
      counts[emoji] = Math.max(0, counts[emoji] - 1);
      button.classList.remove('reacted');
    } else {
      // Add reaction
      userReactions.push(emoji);
      counts[emoji] += 1;
      button.classList.add('reacted');
      
      // Add animation
      button.classList.add('reaction-animate');
      setTimeout(() => button.classList.remove('reaction-animate'), 300);
    }

    // Update stored data
    data[this.currentPage] = {
      userReactions: userReactions,
      counts: counts,
      lastUpdated: Date.now()
    };

    this.saveData(data);
    this.updateDisplay();
  }

  getStoredData() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      return stored ? JSON.parse(stored) : {};
    } catch (e) {
      console.warn('Failed to load reaction data:', e);
      return {};
    }
  }

  saveData(data) {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(data));
    } catch (e) {
      console.warn('Failed to save reaction data:', e);
    }
  }

  loadReactions() {
    const data = this.getStoredData();
    const pageData = data[this.currentPage];
    
    if (pageData) {
      const userReactions = pageData.userReactions || [];
      const counts = pageData.counts || {};

      // Update button states and counts
      const buttons = document.querySelectorAll('.reaction-btn');
      buttons.forEach(button => {
        const emoji = button.dataset.emoji;
        const count = counts[emoji] || 0;
        const countSpan = button.querySelector('.count');
        
        countSpan.textContent = count;
        
        if (userReactions.includes(emoji)) {
          button.classList.add('reacted');
        }

        // Hide count if zero
        if (count === 0) {
          countSpan.style.display = 'none';
        } else {
          countSpan.style.display = 'inline';
        }
      });
    }
  }

  updateDisplay() {
    this.loadReactions();
  }

  // Method to get reaction statistics (for potential future use)
  getStats() {
    const data = this.getStoredData();
    const pageData = data[this.currentPage];
    
    if (!pageData) return null;

    const counts = pageData.counts || {};
    const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
    
    return {
      totalReactions: total,
      reactionCounts: counts,
      mostPopular: Object.entries(counts).sort(([,a], [,b]) => b - a)[0]
    };
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new BlogReactions();
});

// Also initialize if DOM is already loaded (in case script loads late)
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new BlogReactions());
} else {
  new BlogReactions();
}
