// Webcam helper utilities
// Exposes: setupCamera, captureFrame, stopCamera, listVideoDevices

async function listVideoDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter(d => d.kind === 'videoinput');
}

async function setupCamera(videoId, options = {}) {
  const video = document.getElementById(videoId);
  if (!video) throw new Error('Video element not found: ' + videoId);

  const {
    width = 640,
    height = 480,
    facingMode = 'user', // 'user' or 'environment'
    deviceId = undefined, // preferred deviceId
  } = options;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Camera not supported in this browser.');
  }

  // Build constraints
  const constraints = {
    audio: false,
    video: deviceId ? { deviceId: { exact: deviceId } } : { width, height, facingMode }
  };

  // Stop any existing stream first
  try { stopCamera(videoId); } catch (e) {}

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    // Retry with looser constraints
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (e2) {
      const msg = 'Could not access camera. Check permissions and that you are using http://localhost or https.';
      console.error(msg, err, e2);
      throw new Error(msg);
    }
  }

  video.srcObject = stream;

  // Ensure metadata is loaded so dimensions are valid
  await new Promise((resolve) => {
    if (video.readyState >= 1) return resolve(); // HAVE_METADATA
    const onLoaded = () => { video.removeEventListener('loadedmetadata', onLoaded); resolve(); };
    video.addEventListener('loadedmetadata', onLoaded);
  });

  // Attempt autoplay; some browsers require a user gesture
  try {
    await video.play();
  } catch (e) {
    console.warn('Autoplay prevented; waiting for user interaction.', e);
    const onInteract = async () => {
      document.removeEventListener('click', onInteract);
      document.removeEventListener('keydown', onInteract);
      try { await video.play(); } catch (e2) { console.error('Video play failed after interaction', e2); }
    };
    document.addEventListener('click', onInteract);
    document.addEventListener('keydown', onInteract);
  }

  return stream;
}

async function captureFrame(videoId, canvasId, type = 'image/jpeg', quality = 0.85) {
  const video = document.getElementById(videoId);
  const canvas = document.getElementById(canvasId);
  if (!video) throw new Error('Video element not found');
  if (!canvas) throw new Error('Canvas element not found');

  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, type, quality));
  if (!blob) throw new Error('Failed to capture frame');
  return blob;
}

function stopCamera(videoId) {
  const video = document.getElementById(videoId);
  if (!video) return;
  const stream = video.srcObject;
  if (stream && typeof stream.getTracks === 'function') {
    stream.getTracks().forEach(t => t.stop());
  }
  video.srcObject = null;
}

// Expose to global scope for inline scripts in templates
window.setupCamera = setupCamera;
window.captureFrame = captureFrame;
window.stopCamera = stopCamera;
window.listVideoDevices = listVideoDevices;
