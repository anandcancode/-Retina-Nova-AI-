/**
 * RetinaNova AI - Diabetic Retinopathy Diagnostic Platform
 * Frontend Interactions, File Upload Drag-and-Drop, Image Previews & UI Handlers
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Mobile Navigation Toggle
  const mobileToggle = document.getElementById("mobileToggle");
  const navMenu = document.getElementById("navMenu");
  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener("click", function () {
      navMenu.classList.toggle("open");
    });
  }

  // 2. Auto-close flash alerts after 6 seconds
  const alerts = document.querySelectorAll(".alert");
  alerts.forEach(function (alert) {
    const closeBtn = alert.querySelector(".alert-close");
    if (closeBtn) {
      closeBtn.addEventListener("click", function () {
        alert.style.display = "none";
      });
    }
    setTimeout(function () {
      alert.style.transition = "opacity 0.4s ease";
      alert.style.opacity = "0";
      setTimeout(function () {
        alert.remove();
      }, 400);
    }, 7000);
  });

  // 3. Image Drag and Drop & File Upload Preview
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fundusImageInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const previewSize = document.getElementById("previewSize");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const uploadForm = document.getElementById("uploadForm");
  const loadingOverlay = document.getElementById("loadingOverlay");

  if (dropzone && fileInput) {
    // Click on dropzone triggers input
    dropzone.addEventListener("click", function (e) {
      if (e.target !== fileInput) {
        fileInput.click();
      }
    });

    // Drag over & leave
    ["dragenter", "dragover"].forEach(eventName => {
      dropzone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropzone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    // Handle dropped files
    dropzone.addEventListener("drop", function (e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        fileInput.files = files;
        handleFileSelection(files[0]);
      }
    });

    // Handle manual file selection
    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    // Validate file type
    const validExtensions = ["image/jpeg", "image/jpg", "image/png"];
    if (!validExtensions.includes(file.type)) {
      alert("Invalid format! Please upload a JPG, JPEG, or PNG retinal fundus image.");
      if (fileInput) fileInput.value = "";
      return;
    }

    // Validate size (max 15 MB)
    const maxBytes = 15 * 1024 * 1024;
    if (file.size > maxBytes) {
      alert("File is too large! Maximum allowed image size is 15 MB.");
      if (fileInput) fileInput.value = "";
      return;
    }

    // Update preview
    const reader = new FileReader();
    reader.onload = function (e) {
      if (previewImg) previewImg.src = e.target.result;
      if (previewName) previewName.textContent = file.name;
      if (previewSize) {
        const sizeKb = (file.size / 1024).toFixed(1);
        previewSize.textContent = `${sizeKb} KB`;
      }
      if (previewContainer) previewContainer.style.display = "block";
      if (analyzeBtn) analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Handle form submission and show loading spinner
  if (uploadForm) {
    uploadForm.addEventListener("submit", function (e) {
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert("Please select or drop a retinal fundus image before submitting.");
        return;
      }
      if (loadingOverlay) {
        loadingOverlay.style.display = "flex";
      }
      if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing Fundus with Hybrid AI...';
        analyzeBtn.disabled = true;
      }
    });
  }
});

