/**
 * Blog Post Emoji Reactions System
 * Provides Slack-style emoji reactions for blog posts
 * Uses localStorage for user reactions + simulated aggregate data
 */

class BlogReactions {
  constructor() {
    this.defaultReactions = ['👍', '❤️', '😄', '🎉', '🤔', '👏', '🔥', '💡'];
    this.availableEmojis = [
      // Faces & Emotions
      '😀', '😃', '😄', '😁', '😆', '😅', '🤣', '😂', '🙂', '🙃', '😉', '😊', '😇',
      '🥰', '😍', '🤩', '😘', '😗', '☺️', '😚', '😙', '🥲', '😋', '😛', '😜', '🤪',
      '😝', '🤑', '🤗', '🤭', '🤫', '🤔', '🤐', '🤨', '😐', '😑', '😶', '😏', '😒',
      '🙄', '😬', '🤥', '😔', '😪', '🤤', '😴', '😷', '🤒', '🤕', '🤢', '🤮', '🤧',
      '🥵', '🥶', '🥴', '😵', '🤯', '🤠', '🥳', '🥸', '😎', '🤓', '🧐', '😕', '😟',
      '🙁', '☹️', '😮', '😯', '😲', '😳', '🥺', '😦', '😧', '😨', '😰', '😥', '😢',
      '😭', '😱', '😖', '😣', '😞', '😓', '😩', '😫', '🥱', '😤', '😡', '😠', '🤬',
      
      // Hands & Gestures
      '👍', '👎', '👏', '🙌', '👐', '🤲', '🤝', '🙏', '✊', '👊', '🤛', '🤜', '💪',
      '🦾', '🖕', '✌️', '🤞', '🤟', '🤘', '🤙', '👈', '👉', '👆', '🖐️', '✋', '👌',
      '🤏', '👋', '🤚', '🖖', '🤌', '👇', '☝️', '✍️', '💅', '🤳',
      
      // Body Parts
      '🦿', '🦵', '🦶', '👂', '🦻', '👃', '🧠', '🫀', '🫁', '🦷', '🦴', '👀', '👁️', '👅', '👄', '💋', '🩸',
      
      // Symbols & Objects
      '💯', '💢', '💥', '💫', '💦', '💨', '🕳️', '💣', '💬', '👁️‍🗨️', '🗨️', '🗯️', '💭', '💤',
      
      // Transportation
      '🚗', '🚕', '🚙', '🚌', '🚎', '🏎️', '🚓', '🚑', '🚒', '🚐', '🛻', '🚚', '🚛', '🚜', '🏍️', '🛵',
      '🚲', '🛴', '🛹', '🛼', '🚁', '🛸', '✈️', '🛩️', '🛫', '🛬', '🪂', '💺', '🚀', '🛰️', '🚢', '⛵',
      '🚤', '🛥️', '🛳️', '⛴️', '🚂', '🚃', '🚄', '🚅', '🚆', '🚇', '🚈', '🚉', '🚊', '🚝', '🚞', '🚋',
      '🚌', '🚍', '🚘', '🚖', '🚡', '🚠', '🚟', '🎢', '🎡', '🎠', '🏗️', '🚧', '⛽', '🚨', '🚥', '🚦',
      '🛑', '🚏', '⚓', '🛟', '⛵', '🏁', '🚩', '🎌',
      
      // Country Flags
      '🇦🇩', '🇦🇪', '🇦🇫', '🇦🇬', '🇦🇮', '🇦🇱', '🇦🇲', '🇦🇴', '🇦🇶', '🇦🇷', '🇦🇸', '🇦🇹', '🇦🇺', '🇦🇼', '🇦🇽', '🇦🇿',
      '🇧🇦', '🇧🇧', '🇧🇩', '🇧🇪', '🇧🇫', '🇧🇬', '🇧🇭', '🇧🇮', '🇧🇯', '🇧🇱', '🇧🇲', '🇧🇳', '🇧🇴', '🇧🇶', '🇧🇷', '🇧🇸',
      '🇧🇹', '🇧🇻', '🇧🇼', '🇧🇾', '🇧🇿', '🇨🇦', '🇨🇨', '🇨🇩', '🇨🇫', '🇨🇬', '🇨🇭', '🇨🇮', '🇨🇰', '🇨🇱', '🇨🇲', '🇨🇳',
      '🇨🇴', '🇨🇵', '🇨🇷', '🇨🇺', '🇨🇻', '🇨🇼', '🇨🇽', '🇨🇾', '🇨🇿', '🇩🇪', '🇩🇬', '🇩🇯', '🇩🇰', '🇩🇲', '🇩🇴', '🇩🇿',
      '🇪🇦', '🇪🇨', '🇪🇪', '🇪🇬', '🇪🇭', '🇪🇷', '🇪🇸', '🇪🇹', '🇪🇺', '🇫🇮', '🇫🇯', '🇫🇰', '🇫🇲', '🇫🇴', '🇫🇷', '🇬🇦',
      '🇬🇧', '🇬🇩', '🇬🇪', '🇬🇫', '🇬🇬', '🇬🇭', '🇬🇮', '🇬🇱', '🇬🇲', '🇬🇳', '🇬🇵', '🇬🇶', '🇬🇷', '🇬🇸', '🇬🇹', '🇬🇺',
      '🇬🇼', '🇬🇾', '🇭🇰', '🇭🇲', '🇭🇳', '🇭🇷', '🇭🇹', '🇭🇺', '🇮🇨', '🇮🇩', '🇮🇪', '🇮🇱', '🇮🇲', '🇮🇳', '🇮🇴', '🇮🇶',
      '🇮🇷', '🇮🇸', '🇮🇹', '🇯🇪', '🇯🇲', '🇯🇴', '🇯🇵', '🇰🇪', '🇰🇬', '🇰🇭', '🇰🇮', '🇰🇲', '🇰🇳', '🇰🇵', '🇰🇷', '🇰🇼',
      '🇰🇾', '🇰🇿', '🇱🇦', '🇱🇧', '🇱🇨', '🇱🇮', '🇱🇰', '🇱🇷', '🇱🇸', '🇱🇹', '🇱🇺', '🇱🇻', '🇱🇾', '🇲🇦', '🇲🇨', '🇲🇩',
      '🇲🇪', '🇲🇫', '🇲🇬', '🇲🇭', '🇲🇰', '🇲🇱', '🇲🇲', '🇲🇳', '🇲🇴', '🇲🇵', '🇲🇶', '🇲🇷', '🇲🇸', '🇲🇹', '🇲🇺', '🇲🇻',
      '🇲🇼', '🇲🇽', '🇲🇾', '🇲🇿', '🇳🇦', '🇳🇨', '🇳🇪', '🇳🇫', '🇳🇬', '🇳🇮', '🇳🇱', '🇳🇴', '🇳🇵', '🇳🇷', '🇳🇺', '🇳🇿',
      '🇴🇲', '🇵🇦', '🇵🇪', '🇵🇫', '🇵🇬', '🇵🇭', '🇵🇰', '🇵🇱', '🇵🇲', '🇵🇳', '🇵🇷', '🇵🇸', '🇵🇹', '🇵🇼', '🇵🇾', '🇶🇦',
      '🇷🇪', '🇷🇴', '🇷🇸', '🇷🇺', '🇷🇼', '🇸🇦', '🇸🇧', '🇸🇨', '🇸🇩', '🇸🇪', '🇸🇬', '🇸🇭', '🇸🇮', '🇸🇯', '🇸🇰', '🇸🇱',
      '🇸🇲', '🇸🇳', '🇸🇴', '🇸🇷', '🇸🇸', '🇸🇹', '🇸🇻', '🇸🇽', '🇸🇾', '🇸🇿', '🇹🇦', '🇹🇨', '🇹🇩', '🇹🇫', '🇹🇬', '🇹🇭',
      '🇹🇯', '🇹🇰', '🇹🇱', '🇹🇲', '🇹🇳', '🇹🇴', '🇹🇷', '🇹🇹', '🇹🇻', '🇹🇼', '🇹🇿', '🇺🇦', '🇺🇬', '🇺🇲', '🇺🇳', '🇺🇸',
      '🇺🇾', '🇺🇿', '🇻🇦', '🇻🇨', '🇻🇪', '🇻🇬', '🇻🇮', '🇻🇳', '🇻🇺', '🇼🇫', '🇼🇸', '🇽🇰', '🇾🇪', '🇾🇹', '🇿🇦', '🇿🇲', '🇿🇼'
    ];
    this.storageKey = 'blog-reactions';
    this.aggregateStorageKey = 'blog-reactions-aggregate';
    this.customEmojisKey = 'blog-custom-emojis';
    this.currentPage = this.getCurrentPageId();
    this.aggregateData = {};
    this.customEmojis = [];
    this.init();
  }

  getCurrentPageId() {
    // Use the current page path as a unique identifier
    return window.location.pathname;
  }

  init() {
    // Only initialize if we're on a blog post page and not already initialized
    if (this.isBlogPost() && !document.querySelector('.blog-reactions')) {
      this.loadCustomEmojis();
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
    
    const allReactions = [...this.defaultReactions, ...this.customEmojis];
    
    container.innerHTML = `
      <div class="reactions-header">
        <h4>👋 What did you think of this post?</h4>
      </div>
      <div class="reactions-buttons">
        ${allReactions.map(emoji => `
          <button class="reaction-btn" data-emoji="${emoji}">
            <span class="emoji">${emoji}</span>
            <span class="count">0</span>
          </button>
        `).join('')}
        <button class="emoji-picker-btn" title="Add more reactions">
          <span class="emoji">➕</span>
        </button>
      </div>
      <div class="emoji-picker" style="display: none;">
        <div class="emoji-picker-header">
          <span>Pick an emoji:</span>
          <button class="emoji-picker-close">✕</button>
        </div>
        <div class="emoji-picker-grid">
          ${this.availableEmojis.map(emoji => `
            <button class="emoji-option" data-emoji="${emoji}">${emoji}</button>
          `).join('')}
        </div>
      </div>
      <div class="reactions-footer">
        <small>Click to react • Each click adds +1 • ➕ to add more emojis</small>
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
    // Bind reaction buttons
    const buttons = document.querySelectorAll('.reaction-btn');
    buttons.forEach(button => {
      button.addEventListener('click', (e) => {
        e.preventDefault();
        const emoji = button.dataset.emoji;
        this.addReaction(emoji, button);
      });
    });

    // Bind emoji picker button
    const pickerBtn = document.querySelector('.emoji-picker-btn');
    if (pickerBtn) {
      pickerBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.toggleEmojiPicker();
      });
    }

    // Bind emoji picker close button
    const closeBtn = document.querySelector('.emoji-picker-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.hideEmojiPicker();
      });
    }

    // Bind emoji options
    const emojiOptions = document.querySelectorAll('.emoji-option');
    emojiOptions.forEach(option => {
      option.addEventListener('click', (e) => {
        e.preventDefault();
        const emoji = option.dataset.emoji;
        this.selectEmoji(emoji);
      });
    });

    // Close picker when clicking outside
    document.addEventListener('click', (e) => {
      const picker = document.querySelector('.emoji-picker');
      const pickerBtn = document.querySelector('.emoji-picker-btn');
      
      if (picker && picker.style.display !== 'none' && 
          !picker.contains(e.target) && 
          !pickerBtn.contains(e.target)) {
        this.hideEmojiPicker();
      }
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

  loadCustomEmojis() {
    try {
      const stored = localStorage.getItem(this.customEmojisKey);
      this.customEmojis = stored ? JSON.parse(stored) : [];
    } catch (e) {
      console.warn('Failed to load custom emojis:', e);
      this.customEmojis = [];
    }
  }

  saveCustomEmojis() {
    try {
      localStorage.setItem(this.customEmojisKey, JSON.stringify(this.customEmojis));
    } catch (e) {
      console.warn('Failed to save custom emojis:', e);
    }
  }

  loadAggregateData() {
    try {
      const stored = localStorage.getItem(this.aggregateStorageKey);
      this.aggregateData = stored ? JSON.parse(stored) : {};
      
      // Initialize page data if it doesn't exist
      if (!this.aggregateData[this.currentPage]) {
        this.aggregateData[this.currentPage] = {};
        const allReactions = [...this.defaultReactions, ...this.customEmojis];
        allReactions.forEach(emoji => {
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

  toggleEmojiPicker() {
    const picker = document.querySelector('.emoji-picker');
    if (picker) {
      const isVisible = picker.style.display !== 'none';
      picker.style.display = isVisible ? 'none' : 'block';
    }
  }

  hideEmojiPicker() {
    const picker = document.querySelector('.emoji-picker');
    if (picker) {
      picker.style.display = 'none';
    }
  }

  selectEmoji(emoji) {
    // Add emoji to custom reactions if not already present
    const allReactions = [...this.defaultReactions, ...this.customEmojis];
    if (!allReactions.includes(emoji)) {
      this.customEmojis.push(emoji);
      this.saveCustomEmojis();
      
      // Initialize in aggregate data
      if (!this.aggregateData[this.currentPage]) {
        this.aggregateData[this.currentPage] = {};
      }
      if (!this.aggregateData[this.currentPage][emoji]) {
        this.aggregateData[this.currentPage][emoji] = 0;
      }
      
      // Add new reaction button
      this.addReactionButton(emoji);
    }
    
    // Find the button and trigger reaction
    const button = document.querySelector(`[data-emoji="${emoji}"]`);
    if (button && button.classList.contains('reaction-btn')) {
      this.addReaction(emoji, button);
    }
    
    // Hide picker
    this.hideEmojiPicker();
  }

  addReactionButton(emoji) {
    const buttonsContainer = document.querySelector('.reactions-buttons');
    const pickerBtn = document.querySelector('.emoji-picker-btn');
    
    if (buttonsContainer && pickerBtn) {
      // Create new button
      const newButton = document.createElement('button');
      newButton.className = 'reaction-btn';
      newButton.dataset.emoji = emoji;
      newButton.innerHTML = `
        <span class="emoji">${emoji}</span>
        <span class="count">0</span>
      `;
      
      // Insert before the picker button
      buttonsContainer.insertBefore(newButton, pickerBtn);
      
      // Bind event
      newButton.addEventListener('click', (e) => {
        e.preventDefault();
        this.addReaction(emoji, newButton);
      });
    }
  }

  // Method to get reaction statistics (for potential future use)
  getStats() {
    const pageData = this.aggregateData[this.currentPage];
    
    if (!pageData) return null;

    const total = Object.values(pageData).reduce((sum, count) => sum + count, 0);
    
    return {
      totalReactions: total,
      reactionCounts: pageData,
      mostPopular: Object.entries(pageData).sort(([,a], [,b]) => b - a)[0]
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
