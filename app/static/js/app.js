document.addEventListener('DOMContentLoaded', () => {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const panelFile = document.getElementById('panel-file');
  const panelText = document.getElementById('panel-text');
  const fileInput = document.getElementById('file-input');
  const textInput = document.getElementById('text-input');
  const form = document.getElementById('scan-form');
  const scanOverlay = document.getElementById('scan-overlay');
  const popupOverlay = document.getElementById('popup-overlay');
  const popupTitle = document.getElementById('popup-title');
  const popupMessage = document.getElementById('popup-message');
  const popupOk = document.getElementById('popup-ok');
  const toastContainer = document.getElementById('toast-container');

  // Tab switching
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      if (tab === 'file') {
        panelFile?.classList.remove('hidden');
        panelText?.classList.add('hidden');
        if (textInput) textInput.removeAttribute('required');
      } else {
        panelFile?.classList.add('hidden');
        panelText?.classList.remove('hidden');
        if (fileInput) fileInput.value = '';
        const fileName = document.getElementById('file-name');
        if (fileName) fileName.classList.add('hidden');
      }
    });
  });

  // Drag and drop
  const dropzone = document.getElementById('dropzone');
  const fileNameEl = document.getElementById('file-name');

  if (dropzone && fileInput) {
    ['dragenter', 'dragover'].forEach(evt => {
      dropzone.addEventListener(evt, e => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(evt => {
      dropzone.addEventListener(evt, e => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', e => {
      const files = e.dataTransfer?.files;
      if (files?.length) {
        fileInput.files = files;
        showFileName(files[0].name);
        showToast('File ready to scan');
      }
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files?.length) {
        showFileName(fileInput.files[0].name);
        showToast('File selected');
      }
    });
  }

  function showFileName(name) {
    if (fileNameEl) {
      fileNameEl.textContent = name;
      fileNameEl.classList.remove('hidden');
    }
  }

  // Form submit
  form?.addEventListener('submit', e => {
    const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
    if (activeTab === 'text') {
      const text = textInput?.value.trim();
      if (!text) {
        e.preventDefault();
        textInput?.focus();
        showPopup('Paste some text', 'Add at least a few words before running the check.');
        return;
      }
    } else if (activeTab === 'file') {
      if (!fileInput?.files?.length) {
        e.preventDefault();
        dropzone?.classList.add('dragover');
        setTimeout(() => dropzone?.classList.remove('dragover'), 600);
        showPopup('No file selected', 'Drop a file or click the upload area to choose one.');
        return;
      }
    }
    const btn = document.getElementById('submit-btn');
    btn?.classList.add('loading');
    btn?.setAttribute('disabled', 'true');
    openOverlay(scanOverlay);
  });

  // Demo sample forms — show scanning overlay
  document.querySelectorAll('.demo-list form').forEach(demoForm => {
    demoForm.addEventListener('submit', () => openOverlay(scanOverlay));
  });

  // Confirm actions with popup
  document.querySelectorAll('form[data-confirm]').forEach(form => {
    form.addEventListener('submit', e => {
      if (form.dataset.confirmed === '1') {
        delete form.dataset.confirmed;
        return;
      }
      e.preventDefault();
      showPopup('Are you sure?', form.dataset.confirm, () => {
        form.dataset.confirmed = '1';
        form.requestSubmit();
      });
    });
  });

  document.querySelectorAll('[data-copy-link]').forEach(btn => {
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(window.location.href).then(() => {
        showToast('Link copied to clipboard');
      }).catch(() => {
        showPopup('Could not copy', 'Your browser blocked clipboard access.');
      });
    });
  });

  let popupCallback = null;

  function openOverlay(el) {
    if (!el) return;
    el.classList.add('is-open');
    el.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
  }

  function closeOverlay(el) {
    if (!el) return;
    el.classList.remove('is-open');
    el.setAttribute('aria-hidden', 'true');
    if (!document.querySelector('.overlay.is-open')) {
      document.body.classList.remove('modal-open');
    }
  }

  function showPopup(title, message, onOk) {
    if (popupTitle) popupTitle.textContent = title;
    if (popupMessage) popupMessage.textContent = message;
    popupCallback = onOk || null;
    openOverlay(popupOverlay);
  }

  popupOk?.addEventListener('click', () => {
    closeOverlay(popupOverlay);
    if (popupCallback) {
      popupCallback();
      popupCallback = null;
    }
  });

  popupOverlay?.addEventListener('click', e => {
    if (e.target === popupOverlay) closeOverlay(popupOverlay);
  });

  // Toast
  function showToast(message, duration = 2800) {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('is-out');
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  // Stagger signal bar animations on result page
  document.querySelectorAll('.signal-row').forEach((row, i) => {
    row.style.animationDelay = `${0.1 + i * 0.06}s`;
  });

  document.querySelectorAll('.findings-list li').forEach((li, i) => {
    li.style.animationDelay = `${0.1 + i * 0.05}s`;
  });

  // Expose toast for inline scripts
  window.findai = { showToast, showPopup };
});
