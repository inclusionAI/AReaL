document.addEventListener('DOMContentLoaded', function () {
  var path = window.location.pathname || '';
  var isZh = path.indexOf('/zh/') !== -1;
  var isEn = path.indexOf('/en/') !== -1;
  if (!isZh && !isEn) return;

  var currentLang = isZh ? 'zh' : 'en';

  function buildSwitcher() {
    var container = document.createElement('div');
    container.className = 'lang-switcher';

    var select = document.createElement('select');
    select.setAttribute('aria-label', 'Language');
    select.className = 'lang-switcher__select';

    var optionZh = document.createElement('option');
    optionZh.value = 'zh';
    optionZh.textContent = 'Simplified Chinese';
    optionZh.selected = isZh;

    var optionEn = document.createElement('option');
    optionEn.value = 'en';
    optionEn.textContent = 'English';
    optionEn.selected = isEn;

    select.appendChild(optionZh);
    select.appendChild(optionEn);

    select.addEventListener('change', function () {
      var nextLang = select.value;
      if (nextLang === currentLang) return;
      var targetUrl = path.replace('/' + currentLang + '/', '/' + nextLang + '/');
      window.location.href = targetUrl + window.location.search + window.location.hash;
    });

    container.appendChild(select);
    return container;
  }

  function hideOtherLanguageToc() {
    var captions = document.querySelectorAll('p.caption');
    for (var i = 0; i < captions.length; i++) {
      var caption = captions[i];
      var textEl = caption.querySelector('.caption-text');
      if (!textEl) continue;
      var label = (textEl.textContent || '').trim().toLowerCase();

      var isCaptionZh = label === '中文' || label === 'chinese' || label === 'zh';
      var isCaptionEn = label === 'english' || label === 'en';

      if (!isCaptionZh && !isCaptionEn) continue;

      var shouldHide = (currentLang === 'zh' && isCaptionEn) || (currentLang === 'en' && isCaptionZh);
      var shouldHideCaption = true;

      var next = caption.nextElementSibling;
      if (next && next.tagName && next.tagName.toLowerCase() === 'ul') {
        if (shouldHide) {
          caption.style.display = 'none';
          next.style.display = 'none';
        } else if (shouldHideCaption) {
          caption.style.display = 'none';
        }
      } else if (shouldHide) {
        caption.style.display = 'none';
      } else if (shouldHideCaption) {
        caption.style.display = 'none';
      }
    }
  }

  var side = document.querySelector('.wy-side-nav-search');
  if (side) {
    var sideSwitcher = buildSwitcher();
    sideSwitcher.style.marginTop = '8px';
    sideSwitcher.style.display = 'flex';
    sideSwitcher.style.justifyContent = 'center';
    side.appendChild(sideSwitcher);
  } else {
    var topRight = buildSwitcher();
    topRight.style.position = 'fixed';
    topRight.style.top = '12px';
    topRight.style.right = '12px';
    topRight.style.zIndex = '9999';
    document.body.appendChild(topRight);
  }

  hideOtherLanguageToc();
  window.addEventListener('load', hideOtherLanguageToc);
  setTimeout(hideOtherLanguageToc, 50);
  setTimeout(hideOtherLanguageToc, 300);
});
