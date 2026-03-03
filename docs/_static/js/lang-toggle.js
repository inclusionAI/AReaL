/**
 * AReaL Documentation Language Toggle
 * Provides EN/ZH language switching with localStorage preference
 * Implementation differs from slime to maintain uniqueness
 */

(function() {
  'use strict';

  var STORAGE_KEY = 'areal-doc-lang';
  var LANG_EN = 'en';
  var LANG_ZH = 'zh';

  /**
   * Extract language from current URL path
   * Supports: /en/, /zh/, /AReaL/en/, /AReaL/zh/
   */
  function getCurrentLanguage() {
    var path = window.location.pathname;
    var segments = path.split('/').filter(Boolean);

    // Check for language in first or second position
    if (segments[0] === LANG_EN || segments[0] === LANG_ZH) {
      return segments[0];
    }
    if (segments[1] === LANG_EN || segments[1] === LANG_ZH) {
      return segments[1];
    }
    return null;
  }

  /**
   * Determine repository root from URL
   * Returns: 'AReaL' for project site, or null for user site
   */
  function getRepoRoot() {
    var host = window.location.host;
    var segments = window.location.pathname.split('/').filter(Boolean);

    // GitHub Pages project site: ends with github.io and has repo name
    if (host.endsWith('github.io') && segments.length > 0) {
      // Check if first segment could be repo name (not 'en' or 'zh')
      if (segments[0] !== LANG_EN && segments[0] !== LANG_ZH) {
        return segments[0];
      }
    }
    return null;
  }

  /**
   * Build target URL for language switch
   */
  function buildTargetUrl(targetLang) {
    var currentLang = getCurrentLanguage();
    var repoRoot = getRepoRoot();
    var segments = window.location.pathname.split('/').filter(Boolean);
    var hasLang = segments[0] === LANG_EN || segments[0] === LANG_ZH ||
                  segments[1] === LANG_EN || segments[1] === LANG_ZH;

    if (hasLang) {
      // Replace existing language prefix
      if (segments[0] === LANG_EN || segments[0] === LANG_ZH) {
        segments[0] = targetLang;
      } else if (segments[1] === LANG_EN || segments[1] === LANG_ZH) {
        segments[1] = targetLang;
      }
    } else {
      // Insert language prefix
      if (repoRoot) {
        // Project site: insert after repo root
        segments.splice(1, 0, targetLang);
      } else {
        // User site: insert at beginning
        segments.unshift(targetLang);
      }
    }

    var newPath = '/' + segments.join('/') + '/';
    var url = new URL(window.location.href);
    url.pathname = newPath;
    return url.toString();
  }

  /**
   * Save language preference to localStorage
   */
  function savePreference(lang) {
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch (e) {
      // localStorage unavailable
    }
  }

  /**
   * Create and insert language toggle button
   */
  function createButton() {
    var currentLang = getCurrentLanguage();
    if (!currentLang) return;

    var targetLang = currentLang === LANG_EN ? LANG_ZH : LANG_EN;
    var buttonText = currentLang === LANG_EN ? '中文' : 'EN';
    var titleText = currentLang === LANG_EN ? 'Switch to Chinese' : 'Switch to English';

    var button = document.createElement('button');
    button.id = 'areal-lang-toggle';
    button.className = 'areal-lang-toggle-btn';
    button.type = 'button';
    button.textContent = buttonText;
    button.title = titleText;

    button.addEventListener('click', function(e) {
      e.preventDefault();
      var targetUrl = buildTargetUrl(targetLang);
      savePreference(targetLang);
      window.location.href = targetUrl;
    });

    // Find appropriate container
    var container = document.querySelector('.bd-navbar') ||
                    document.querySelector('.header-article-items__end') ||
                    document.querySelector('nav') ||
                    document.body;

    // Try to append to navbar end
    var navbar = document.querySelector('.bd-navbar');
    if (navbar) {
      navbar.appendChild(button);
    } else {
      container.appendChild(button);
    }
  }

  /**
   * Initialize on page load
   */
  function init() {
    if (!getCurrentLanguage()) return;

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', createButton);
    } else {
      createButton();
    }
  }

  init();
})();
