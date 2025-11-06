/**
 * Blog Post Emoji Reactions System
 * Provides Slack-style emoji reactions for blog posts
 * Uses localStorage for user reactions + simulated aggregate data
 */

class BlogReactions {
  constructor() {
    this.reactions = ['👍', '❤️', '😄', '🎉', '🤔', '👏', '🔥', '💡'];
    this.storageKey = 'blog-reactions';
    this.aggregateStorageKey = 'blog-reactions-aggregate';
    this.currentPage = this.getCurrentPageId();
    this.aggregateData = {};
    this.init();
  }

  getCurrentPageId() {
    // Use the current page path as a unique identifier
    return window.location.pathname;
  }

  init() {
    // Only initialize if we're on a blog post page and not already initialized
    if (this.isBlogPost() && !document.querySelector('.blog-reactions')) {
      this.createReactionContainer();
      this.loadAggregateData();
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
        <small>Click to react • Each click adds +1 • Your clicks are highlighted</small>
      </div>
    `;

    // Insert at the end of the main content section (before any footer)
    const section = document.querySelector('section');
    if (section) {
      section.appendChild(container);
    } else {
      // Fallback: insert before footer but try to stay in main content
      const footer = document.querySelector('footer');
      if (footer) {
        footer.parentNode.insertBefore(container, footer);
      } else {
        document.body.appendChild(container);
      }
    }
  }

  bindEvents() {
    const buttons = document.querySelectorAll('.reaction-btn');
    buttons.forEach(button => {
      button.addEventListener('click', (e) => {
        e.preventDefault();
        const emoji = button.dataset.emoji;
        this.addReaction(emoji, button);
      });
    });
  }

  addReaction(emoji, button) {
    const data = this.getStoredData();
    const pageData = data[this.currentPage] || {};
    const userClicks = pageData.userClicks || {};

    // Initialize user clicks for this emoji if it doesn't exist
    if (!userClicks[emoji]) {
      userClicks[emoji] = 0;
    }

    // Always increment - no toggle behavior
    userClicks[emoji] += 1;
    this.updateAggregateData(emoji, 1);
    
    // Add animation
    button.classList.add('reaction-animate');
    setTimeout(() => button.classList.remove('reaction-animate'), 300);

    // Update stored data
    data[this.currentPage] = {
      userClicks: userClicks,
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

  loadAggregateData() {
    try {
      const stored = localStorage.getItem(this.aggregateStorageKey);
      this.aggregateData = stored ? JSON.parse(stored) : {};
      
      // Initialize page data if it doesn't exist
      if (!this.aggregateData[this.currentPage]) {
        this.aggregateData[this.currentPage] = {};
        this.reactions.forEach(emoji => {
          this.aggregateData[this.currentPage][emoji] = 0;
        });
      }
    } catch (e) {
      console.warn('Failed to load aggregate data:', e);
      this.aggregateData = {};
    }
  }


  updateAggregateData(emoji, delta) {
    if (!this.aggregateData[this.currentPage]) {
      this.aggregateData[this.currentPage] = {};
    }
    
    if (!this.aggregateData[this.currentPage][emoji]) {
      this.aggregateData[this.currentPage][emoji] = 0;
    }
    
    this.aggregateData[this.currentPage][emoji] = Math.max(0, 
      this.aggregateData[this.currentPage][emoji] + delta
    );
    
    // Save aggregate data
    try {
      localStorage.setItem(this.aggregateStorageKey, JSON.stringify(this.aggregateData));
    } catch (e) {
      console.warn('Failed to save aggregate data:', e);
    }
  }

  loadReactions() {
    const data = this.getStoredData();
    const pageData = data[this.currentPage];
    const userClicks = pageData ? pageData.userClicks || {} : {};

    // Update button states and counts
    const buttons = document.querySelectorAll('.reaction-btn');
    buttons.forEach(button => {
      const emoji = button.dataset.emoji;
      const totalCount = this.aggregateData[this.currentPage] ? 
        this.aggregateData[this.currentPage][emoji] || 0 : 0;
      const countSpan = button.querySelector('.count');
      
      countSpan.textContent = totalCount;
      
      // Check if user has clicked this emoji (for visual feedback)
      if (userClicks[emoji] && userClicks[emoji] > 0) {
        button.classList.add('reacted');
      } else {
        button.classList.remove('reacted');
      }

      // Hide count if zero
      if (totalCount === 0) {
        countSpan.style.display = 'none';
      } else {
        countSpan.style.display = 'inline';
      }
    });
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

// Initialize reactions system
let reactionsInstance = null;

function initializeReactions() {
  if (!reactionsInstance) {
    reactionsInstance = new BlogReactions();
  }
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeReactions);
} else {
  initializeReactions();
}
