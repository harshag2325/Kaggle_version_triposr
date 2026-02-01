/* Updated editor.js — includes Generate Shadow button and hybrid shadow generation
   - Uses selected shadow side (auto/front/back/left/right)
   - When user clicks Generate Shadow, it captures 3D renderer passes,
     computes shadow direction using model orientation, and composites.
*/

const canvasEl = document.getElementById("c");
const canvasWrapper = document.getElementById("canvas-wrapper");
const finalImgEl = document.getElementById("final-img");

let canvas = new fabric.Canvas("c", { preserveObjectStacking: true });

let currentBgUrl = null;
let currentFgUrl = null;
let fgObj = null;

let lastModelURL = "";
let lastTextureURL = "";

window.threeScene = null;
window.threeCamera = null;
window.threeRenderer = null;
window.sunLight = null;
window.shadowPlane = null;
window.fabric3D = null;
window.sunControl = null;
window.model3D = null;

let forcedShadowDirection = "auto"; // 'auto'|'front'|'back'|'left'|'right'
function dataURLtoBlob(dataUrl) {
  const arr = dataUrl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) u8arr[n] = bstr.charCodeAt(n);
  return new Blob([u8arr], { type: mime });
}
// safeFetch (unchanged)
async function safeFetch(url, opts = {}) {
    const r = await fetch(url, opts);

    // try JSON
    try {
        const txt = await r.text();
        try {
            return JSON.parse(txt);
        } catch (e) {
            return { raw: txt };   // <-- return raw string
        }
    } catch (err) {
        throw new Error("Fetch failed: " + err.message);
    }
}


// resetCanvas
function resetCanvas() {
  canvas.clear();
  fgObj = null;

  currentBgUrl = null;
  currentFgUrl = null;

  const old = document.getElementById("threeCanvasInFabric");
  if (old) old.remove();

  finalImgEl.style.display = "none";
  finalImgEl.src = "";
}
window.resetCanvas = resetCanvas;

// Upload and loadEditor (kept simplified, same as your original)
async function uploadImages() {
  try {
    const bgFile = document.getElementById("img1").files[0];
    if (!bgFile) return alert("Select background");

    const mode = document.querySelector('input[name="fg_source"]:checked').value;

    if (mode === "file") {
      const fgFile = document.getElementById("img2").files[0];
      if (!fgFile) return alert("Select foreground");

      const f = new FormData();
      f.append("image1", bgFile);
      f.append("image2", fgFile);

      const out = await safeFetch("/upload", { method: "POST", body: f });
      currentBgUrl = out.image1;
      currentFgUrl = out.image2_nobg;

      loadEditor(currentBgUrl, currentFgUrl);
      return;
    }

    const prompt = document.getElementById("fgPrompt").value.trim();
    if (!prompt) return alert("Enter prompt");

    const f1 = new FormData();
    f1.append("image1", bgFile);
    f1.append("image2", bgFile);
    let bgOut = await safeFetch("/upload", { method: "POST", body: f1 });
    currentBgUrl = bgOut.image1;

    const f2 = new FormData();
    f2.append("prompt", prompt);
    f2.append("keep_shadows", document.getElementById("fgShadowChoice").value);

    let fgOut = await safeFetch("/prompt_fg", { method: "POST", body: f2 });
    currentFgUrl = fgOut.fg;

    loadEditor(currentBgUrl, currentFgUrl);
  } catch (err) {
    console.error(err);
    alert("Upload failed");
  }
}
window.uploadImages = uploadImages;

// loadEditor: set background and add fg image
function loadEditor(bgUrl, fgUrl) {
  canvas.clear();

  fabric.Image.fromURL(bgUrl, bg => {
    const sc = Math.min(canvas.width / bg.width, canvas.height / bg.height);

    canvas.setBackgroundImage(bg, canvas.renderAll.bind(canvas), {
      left: canvas.width / 2,
      top: canvas.height / 2,
      originX: "center",
      originY: "center",
      scaleX: sc,
      scaleY: sc,
    });
  });

  fabric.Image.fromURL(fgUrl, img => {
    fgObj = img;

    const maxW = canvas.width * 0.4;
    const maxH = canvas.height * 0.4;
    const sc = Math.min(maxW / img.width, maxH / img.height);

    img.scale(sc);
    img.set({
      left: canvas.width / 2,
      top: canvas.height / 2,
      originX: "center",
      originY: "center",
      hasControls: true,
      hasBorders: true,
      borderColor: "#222",
      cornerColor: "#111",
      cornerStrokeColor: "#fff",
      cornerSize: 15
    });

    canvas.add(img);
    canvas.setActiveObject(img);
    img.setCoords();
    canvas.renderAll();
  });
}

// blendOnly (unchanged)
async function blendOnly() {
  if (!fgObj) return alert("No FG");

  const w = Math.round(fgObj.getScaledWidth());
  const h = Math.round(fgObj.getScaledHeight());
  const x = Math.round(fgObj.left - w / 2);
  const y = Math.round(fgObj.top - h / 2);

  const f = new FormData();
  f.append("bg", currentBgUrl);
  f.append("fg", currentFgUrl);
  f.append("x", x);
  f.append("y", y);
  f.append("w", w);
  f.append("h", h);
  f.append("canvas_w", canvas.width);
  f.append("canvas_h", canvas.height);

  const out = await safeFetch("/blend", { method: "POST", body: f });

  finalImgEl.src = out.final;
  finalImgEl.style.display = "block";
}
window.blendOnly = blendOnly;

// generate3DAndUse
async function generate3DAndUse() {
  if (!currentFgUrl) return alert("Upload FG first");

  const f = new FormData();
  f.append("fg", currentFgUrl);

  const out = await safeFetch("/triposr", { method: "POST", body: f });

  lastModelURL = out.model;
  lastTextureURL = out.texture;

  alert("3D ready → Click 'Place Rotatable 3D'");
}
window.generate3DAndUse = generate3DAndUse;

// updateSunFromSunControl (unchanged)
function updateSunFromSunControl(sun) {
  if (!window.sunLight || !window.fabric3D) return;

  const obj = window.fabric3D;

  const centerX = obj.left + obj.getScaledWidth() / 2;
  const centerY = obj.top + obj.getScaledHeight() / 2;

  const dx = sun.left - centerX;
  const dy = sun.top - centerY;

  const angle = Math.atan2(dy, dx);

  const DIST = 4;
  const lx = Math.cos(angle) * DIST;
  const lz = Math.sin(angle) * DIST;

  window.sunLight.position.set(lx, 3, lz);

  if (window.sunLight.target) {
    window.sunLight.target.position.set(0, 0, 0);
    window.sunLight.target.updateMatrixWorld();
  }
}

// renderAndCapture (kept same as original)
function renderAndCapture(options = {}) {
  return new Promise((resolve, reject) => {
    try {
      if (!window.threeRenderer || !window.threeScene || !window.threeCamera) {
        return reject(new Error("Three renderer/scene/camera not initialized"));
      }

      const renderer = window.threeRenderer;
      const scene = window.threeScene;
      const camera = window.threeCamera;

      const overlay = document.getElementById("threeCanvasInFabric");
      const w = overlay.clientWidth;
      const h = overlay.clientHeight;
      renderer.setSize(w, h);

      const modified = [];
      scene.traverse(o => {
        if (o.isMesh) {
          modified.push({
            mesh: o,
            origColorWrite: o.material && o.material.colorWrite !== undefined ? o.material.colorWrite : true,
            origVisible: o.visible
          });
        }
      });

      let shadowPlaneOrigVis = null;
      if (window.shadowPlane) shadowPlaneOrigVis = window.shadowPlane.visible;

      if (options.shadowPass) {
        if (window.shadowPlane) window.shadowPlane.visible = true;
        scene.traverse(o => {
          if (o.isMesh) {
            if (o.material && o.material.colorWrite !== undefined) {
              o.material.colorWrite = false;
            } else if (o.material) {
              o.material.visible = true;
            }
          }
        });
      } else {
        if (window.shadowPlane) window.shadowPlane.visible = false;
        scene.traverse(o => {
          if (o.isMesh) {
            if (o.material && o.material.colorWrite !== undefined) o.material.colorWrite = true;
            o.visible = true;
          }
        });
      }

      const prevClearAlpha = renderer.getClearAlpha ? renderer.getClearAlpha() : 1.0;
      renderer.setClearColor(0x000000, 0);

      renderer.render(scene, camera);

      setTimeout(() => {
        try {
          const dataURL = renderer.domElement.toDataURL("image/png");

          modified.forEach(m => {
            if (m.mesh.material && m.mesh.material.colorWrite !== undefined) {
              m.mesh.material.colorWrite = m.origColorWrite;
            }
            m.mesh.visible = m.origVisible;
          });

          if (window.shadowPlane) window.shadowPlane.visible = shadowPlaneOrigVis;

          try { renderer.setClearColor(0x000000, prevClearAlpha); } catch (e) {}

          resolve(dataURL);

        } catch (err) {

          modified.forEach(m => {
            if (m.mesh.material && m.mesh.material.colorWrite !== undefined) {
              m.mesh.material.colorWrite = m.origColorWrite;
            }
            m.mesh.visible = m.origVisible;
          });
          if (window.shadowPlane) window.shadowPlane.visible = shadowPlaneOrigVis;
          try { renderer.setClearColor(0x000000, prevClearAlpha); } catch (e) {}
          reject(err);
        }
      }, 30);
    } catch (e) {
      reject(e);
    }
  });
}

// Homography utilities (computeHomography, solveLinear, invert3x3) are same as before
function computeHomography(src, dst) {
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const xs = src[i].x, ys = src[i].y;
    const xd = dst[i].x, yd = dst[i].y;

    A.push([xs, ys, 1, 0, 0, 0, -xs * xd, -ys * xd]);
    b.push(xd);

    A.push([0, 0, 0, xs, ys, 1, -xs * yd, -ys * yd]);
    b.push(yd);
  }

  const h = solveLinear(A, b);
  if (!h) return null;

  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1]
  ];
}

function solveLinear(A, b) {
  const n = A.length;
  const m = A[0].length;

  const M = new Array(n);
  for (let i = 0; i < n; i++) {
    M[i] = A[i].slice();
    M[i].push(b[i]);
  }

  const rows = n, cols = m + 1;
  let r = 0;

  for (let c = 0; c < m && r < rows; c++) {
    let piv = r;
    for (let i = r; i < rows; i++) {
      if (Math.abs(M[i][c]) > Math.abs(M[piv][c])) piv = i;
    }
    if (Math.abs(M[piv][c]) < 1e-12) continue;

    let tmp = M[r];
    M[r] = M[piv];
    M[piv] = tmp;

    const div = M[r][c];
    for (let j = c; j < cols; j++) M[r][j] /= div;

    for (let i = 0; i < rows; i++) {
      if (i === r) continue;
      const mul = M[i][c];
      if (Math.abs(mul) < 1e-12) continue;
      for (let j = c; j < cols; j++) M[i][j] -= mul * M[r][j];
    }

    r++;
  }

  const x = new Array(m).fill(0);

  for (let i = 0; i < m; i++) {
    let found = -1;
    for (let rr = 0; rr < rows; rr++) {
      if (Math.abs(M[rr][i] - 1) < 1e-9) {
        found = rr;
        break;
      }
    }

    if (found === -1) return null;

    x[i] = M[found][cols - 1];
  }

  return x;
}

function invert3x3(H) {
  const a = H[0][0], b = H[0][1], c = H[0][2];
  const d = H[1][0], e = H[1][1], f = H[1][2];
  const g = H[2][0], h = H[2][1], i = H[2][2];

  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;

  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);

  const G = b * f - c * e;
  const H0 = -(a * f - c * d);
  const I = a * e - b * d;

  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;

  const invDet = 1 / det;

  return [
    [A * invDet, D * invDet, G * invDet],
    [B * invDet, E * invDet, H0 * invDet],
    [C * invDet, F * invDet, I * invDet]
  ];
}

// getModelFootPointsInOverlay (unchanged)
function getModelFootPointsInOverlay() {
  try {
    const obj = window.model3D;
    const camera = window.threeCamera;
    const renderer = window.threeRenderer;
    const overlay = document.getElementById("threeCanvasInFabric");

    if (!obj || !camera || !renderer || !overlay)
      return [];

    const rd = renderer.domElement.getBoundingClientRect();
    const ov = overlay.getBoundingClientRect();

    const offsetX = ov.left - rd.left;
    const offsetY = ov.top - rd.top;

    const W = renderer.domElement.width;
    const H = renderer.domElement.height;

    const pts = [];
    const v = new THREE.Vector3();

    obj.traverse(c => {
      if (!c.isMesh) return;
      const pos = c.geometry?.attributes?.position;
      if (!pos) return;

      const stride = Math.max(1, Math.floor(pos.count / 4000));

      for (let i = 0; i < pos.count; i += stride) {
        v.fromBufferAttribute(pos, i);
        c.localToWorld(v);
        pts.push(v.clone());
      }
    });

    if (pts.length === 0) return [];

    let minY = Infinity;
    for (const p of pts) if (p.y < minY) minY = p.y;

    const eps = 0.03;
    const feet = pts.filter(p => p.y < minY + eps);


    if (feet.length === 0) return [];

    const result = [];

    for (const p of feet) {
      const ndc = p.clone().project(camera);

      const rx = ((ndc.x + 1) / 2) * W;
      const ry = ((-ndc.y + 1) / 2) * H;

      result.push({
        x: rx - offsetX,
        y: ry - offsetY
      });
    }

    return result;

  } catch (e) {
    console.warn("getModelFootPointsInOverlay failed", e);
    return [];
  }
}

// compositeShadowPerspective: accepts opts.shadowDir {sx,sy} in pixel-space direction (overlay pixel delta)
async function compositeShadowPerspective(
  shadowDataURL,
  objectDataURL,
  overlayW,
  overlayH,
  opts = {}
) {
  opts = Object.assign({
    shadowDownscaleMax: 700,
    blurPx: 18,
    shadowOpacity: 0.55,
    shadowDir: null // optional {sx, sy} in overlay pixel space (not normalized)
  }, opts);

  const loadImg = src => new Promise((res, rej) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => res(im);
    im.onerror = e => rej(e);
    im.src = src;
  });

  const [shadowImg, objImg] = await Promise.all([
    loadImg(shadowDataURL),
    loadImg(objectDataURL)
  ]);

  const scale = Math.min(1, opts.shadowDownscaleMax /
    Math.max(shadowImg.width, shadowImg.height));

  const sw = Math.max(1, Math.round(shadowImg.width * scale));
  const sh = Math.max(1, Math.round(shadowImg.height * scale));

  const tmpCan = document.createElement("canvas");
  tmpCan.width = sw;
  tmpCan.height = sh;

  const tctx = tmpCan.getContext("2d");
  tctx.clearRect(0, 0, sw, sh);
  tctx.drawImage(shadowImg, 0, 0, sw, sh);

  // fabric box
  const obj = window.fabric3D;
  const left = obj.left;
  const top = obj.top;
  const w = obj.getScaledWidth();
  const h = obj.getScaledHeight();

  const tl = { x: left, y: top };
  const tr = { x: left + w, y: top };
  const br = { x: left + w, y: top + h };
  const bl = { x: left, y: top + h };

  // compute box normalized positions
  const boxNorm = pt => ({
    x: (pt.x - left) / w * overlayW,
    y: (pt.y - top) / h * overlayH
  });

  const norm_tl = boxNorm(tl);
  const norm_tr = boxNorm(tr);
  const norm_br = boxNorm(br);
  const norm_bl = boxNorm(bl);

  // compute shadow direction sx,sy (in overlay pixel coordinates)
  let sx = 0, sy = 0;
  if (opts.shadowDir && typeof opts.shadowDir.sx === 'number') {
    sx = opts.shadowDir.sx;
    sy = opts.shadowDir.sy;
  } else {
    // fallback: use sun control position in fabric space
    const sunCtrl = window.sunControl || { left: left + w + 40, top: top + h/2 };
    const boxCenterX = obj.left + w / 2;
    const boxCenterY = obj.top + h / 2;

    const dx = sunCtrl.left - boxCenterX;
    const dy = sunCtrl.top - boxCenterY;

    const ang = Math.atan2(-dy, -dx); // opposite for shadow
    sx = Math.cos(ang);
    sy = Math.sin(ang);

    // convert normalized to overlay pixels
    const len = Math.max(20, Math.max(overlayW, overlayH) * 0.5);
    sx *= len; sy *= len;
  }

  // compute feet anchors
  const feet = getModelFootPointsInOverlay();
  let dstTopLeft, dstTopRight;
  let dstBottomLeft, dstBottomRight;

  const baseLen = Math.max(0.3, Math.min(2.5, (Math.abs(sx) + Math.abs(sy)) / 200));
  const length = baseLen * Math.max(overlayW, overlayH) * 0.8;

  if (feet.length >= 2) {
    feet.sort((a,b)=>a.x - b.x);
    const leftFoot  = feet[0];
    const rightFoot = feet[feet.length - 1];
    const avgFeetY = (leftFoot.y + rightFoot.y) / 2;

    dstTopLeft  = { x: leftFoot.x,  y: avgFeetY };
    dstTopRight = { x: rightFoot.x, y: avgFeetY };

    const extend = Math.max(20, length * 1.15);
    const squash = 0.6;

    dstBottomLeft = {
      x: leftFoot.x + sx * extend / Math.max(1, Math.abs(sx)),
      y: leftFoot.y + sy * extend / Math.max(1, Math.abs(sy)) * squash
    };

    dstBottomRight = {
      x: rightFoot.x + sx * extend / Math.max(1, Math.abs(sx)),
      y: rightFoot.y + sy * extend / Math.max(1, Math.abs(sy)) * squash
    };

  } else {
    const extend = Math.max(20, length * 1.15);
    const squash = 0.6;

    dstTopLeft  = norm_bl;
    dstTopRight = norm_br;

    dstBottomLeft = {
      x: norm_bl.x + sx * extend / Math.max(1, Math.abs(sx)),
      y: norm_bl.y + sy * extend / Math.max(1, Math.abs(sy)) * squash
    };

    dstBottomRight = {
      x: norm_br.x + sx * extend / Math.max(1, Math.abs(sx)),
      y: norm_br.y + sy * extend / Math.max(1, Math.abs(sy)) * squash
    };
  }
  // ↓↓↓ ADD THIS -- pushes the whole shadow down ↓↓↓

  const dstQuadPixel = [
    dstTopLeft,
    dstTopRight,
    dstBottomRight,
    dstBottomLeft
  ];

  // Auto-align shadow vertically so top of shadow matches model feet projection
  try {
    if (feet && feet.length > 0) {
      const modelLowestY = Math.max(...feet.map(f => f.y));
      const shadowTopY = (dstTopLeft.y + dstTopRight.y) / 2;
      const autoOffset = modelLowestY - shadowTopY;
      dstQuadPixel.forEach(p => { p.y += autoOffset; });
    }
  } catch (e) {
    // fallback: no offset
  }

  const dst = dstQuadPixel.map(p => ({
    x: p.x / overlayW,
    y: p.y / overlayH
  }));

  const src = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 }
  ];

  const H = computeHomography(src, dst);
  if (!H) throw new Error("homography compute failed");

  const Hinv = invert3x3(H);
  if (!Hinv) throw new Error("homography inverse failed");

  const scaleMat = [
    [1 / overlayW, 0, 0],
    [0, 1 / overlayH, 0],
    [0, 0, 1]
  ];

  const Hinv_pixels = multiply3x3(Hinv, scaleMat);

  // inverse warp
  const warpedCanvas = document.createElement("canvas");
  warpedCanvas.width = overlayW;
  warpedCanvas.height = overlayH;
  const wctx = warpedCanvas.getContext("2d");

  const warpedImgData = wctx.createImageData(overlayW, overlayH);
  const warpedBuf = warpedImgData.data;

  const sctx2 = tmpCan.getContext("2d");
  const sdata = sctx2.getImageData(0, 0, sw, sh).data;

  function mapDstToSrc(x, y) {
    const xn = x;
    const yn = y;

    const m = Hinv_pixels;

    const u = m[0][0] * xn + m[0][1] * yn + m[0][2];
    const v = m[1][0] * xn + m[1][1] * yn + m[1][2];
    const w0 = m[2][0] * xn + m[2][1] * yn + m[2][2];

    if (Math.abs(w0) < 1e-9) return null;

    const sxn = (u / w0) * (sw - 1);
    const syn = (v / w0) * (sh - 1);

    return { sx: sxn, sy: syn };
  }

  function sampleBilinear(sx, sy) {
    if (sx < 0 || sy < 0 || sx >= sw - 1 || sy >= sh - 1)
      return [0, 0, 0, 0];

    const x0 = Math.floor(sx);
    const y0 = Math.floor(sy);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const dx = sx - x0;
    const dy = sy - y0;

    const idx00 = (y0 * sw + x0) * 4;
    const idx10 = (y0 * sw + x1) * 4;
    const idx01 = (y1 * sw + x0) * 4;
    const idx11 = (y1 * sw + x1) * 4;

    const s00 = sdata.slice(idx00, idx00 + 4);
    const s10 = sdata.slice(idx10, idx10 + 4);
    const s01 = sdata.slice(idx01, idx01 + 4);
    const s11 = sdata.slice(idx11, idx11 + 4);

    const r =
      (s00[0] * (1 - dx) + s10[0] * dx) * (1 - dy) +
      (s01[0] * (1 - dx) + s11[0] * dx) * dy;

    const g =
      (s00[1] * (1 - dx) + s10[1] * dx) * (1 - dy) +
      (s01[1] * (1 - dx) + s11[1] * dx) * dy;

    const b =
      (s00[2] * (1 - dx) + s10[2] * dx) * (1 - dy) +
      (s01[2] * (1 - dx) + s11[2] * dx) * dy;

    const a =
      (s00[3] * (1 - dx) + s10[3] * dx) * (1 - dy) +
      (s01[3] * (1 - dx) + s11[3] * dx) * dy;

    return [r, g, b, a];
  }

  for (let y = 0; y < overlayH; y++) {
    for (let x = 0; x < overlayW; x++) {
      let mapped = mapDstToSrc(x, y);
      const idx = (y * overlayW + x) * 4;

      if (!mapped) {
        warpedBuf[idx] = warpedBuf[idx+1] = warpedBuf[idx+2] = 0;
        warpedBuf[idx+3] = 0;
        continue;
      }

      const s = sampleBilinear(mapped.sx, mapped.sy);

      warpedBuf[idx] = s[0];
      warpedBuf[idx+1] = s[1];
      warpedBuf[idx+2] = s[2];
      warpedBuf[idx+3] = s[3];
    }
  }

  wctx.putImageData(warpedImgData, 0, 0);

  // Blur + composite
  const finalCanvas = document.createElement("canvas");
  finalCanvas.width = overlayW;
  finalCanvas.height = overlayH;
  const fctx = finalCanvas.getContext("2d");

  fctx.filter = `blur(${opts.blurPx}px)`;
  fctx.globalAlpha = opts.shadowOpacity;
  fctx.drawImage(warpedCanvas, 0, 0);

  fctx.filter = "none";
  fctx.globalAlpha = 1.0;
  fctx.drawImage(objImg, 0, 0);

  return finalCanvas.toDataURL("image/png");
}

// captureThenBlend: unchanged but now supports passing shadowDir through composite call
async function captureThenBlend() {
const overlay = document.getElementById("threeCanvasInFabric");
if (!overlay || !window.threeRenderer) return blendOnly();


const overlayW = overlay.clientWidth;
const overlayH = overlay.clientHeight;


try {
const shadowDataURL = await renderAndCapture({ shadowPass: true });
const objectDataURL = await renderAndCapture({ shadowPass: false });


const combinedDataURL = await compositeShadowPerspective(
shadowDataURL,
objectDataURL,
overlayW,
overlayH,
{
shadowDownscaleMax: 700,
blurPx: 18,
shadowOpacity: 0.5
}
);


// FIX HERE → Convert base64 to Blob
const f1 = new FormData();
f1.append("image_data", combinedDataURL); // send raw base64




const up = await safeFetch("/upload_3d_view", {
method: "POST",
body: f1
});


const fgURL = up.fg|| up.raw|| up.url ;


const obj = window.fabric3D;
// CORRECT CODE
const W = Math.round(obj.getScaledWidth());
const H = Math.round(obj.getScaledHeight());
const X = Math.round(obj.left);
const Y = Math.round(obj.top);


const f = new FormData();
f.append("bg", currentBgUrl);
f.append("fg", fgURL);
f.append("x", X);
f.append("y", Y);
f.append("w", W);
f.append("h", H);
f.append("canvas_w", canvas.width);
f.append("canvas_h", canvas.height);


const out = await safeFetch("/blend", { method: "POST", body: f });


finalImgEl.src = out.final;
finalImgEl.style.display = "block";


} catch (err) {
console.error("captureThenBlend failed:", err);
alert("3D blend failed → " + err.message);
}
}
window.captureThenBlend = captureThenBlend;

// NEW: compute overlay pixel direction from a 3D world direction relative to model
function computeOverlayShadowDirFromWorldDir(worldDir) {
  // uses model3D, camera, renderer to map a small step in world to overlay pixel delta
  try {
    const renderer = window.threeRenderer;
    const camera = window.threeCamera;
    const obj = window.model3D;
    if (!renderer || !camera || !obj) return null;

    const rd = renderer.domElement;
    const W = rd.width || rd.clientWidth;
    const H = rd.height || rd.clientHeight;

    const SHADOW_PLANE_Y = window.shadowPlane ? window.shadowPlane.position.y : -0.6;

    const p0 = new THREE.Vector3();
    obj.getWorldPosition(p0);
    p0.y = SHADOW_PLANE_Y;

    const p1 = p0.clone().add(worldDir.clone().normalize().multiplyScalar(0.2));

    const ndc0 = p0.clone().project(camera);
    const ndc1 = p1.clone().project(camera);

    const x0 = ((ndc0.x + 1) / 2) * W;
    const y0 = ((-ndc0.y + 1) / 2) * H;

    const x1 = ((ndc1.x + 1) / 2) * W;
    const y1 = ((-ndc1.y + 1) / 2) * H;

    const sx = x1 - x0;
    const sy = y1 - y0;
    return { sx, sy };
  } catch (e) {
    console.warn("computeOverlayShadowDirFromWorldDir failed", e);
    return null;
  }
}

// NEW: generateShadowNow — main entrypoint for the user pressing "Generate Shadow"
async function generateShadowNow() {
if (!window.model3D || !window.threeRenderer) {
return alert("Place a 3D model first");
}


const overlay = document.getElementById("threeCanvasInFabric");
const overlayW = overlay.clientWidth;
const overlayH = overlay.clientHeight;


try {
const shadowDataURL = await renderAndCapture({ shadowPass: true });
const objectDataURL = await renderAndCapture({ shadowPass: false });


const combinedDataURL = await compositeShadowPerspective(
shadowDataURL,
objectDataURL,
overlayW,
overlayH,
{
shadowDownscaleMax: 700,
blurPx: 18,
shadowOpacity: 0.55
}
);


// FIX HERE → Blob upload
const f1 = new FormData();
f1.append("image_data", combinedDataURL); // send raw base64



const up = await safeFetch("/upload_3d_view", {
method: "POST",
body: f1
});


const fgURL = up.fg|| up.raw|| up.url ;
const obj = window.fabric3D;


const W = Math.round(obj.getScaledWidth());
const H = Math.round(obj.getScaledHeight());
const X = Math.round(obj.left);
const Y = Math.round(obj.top);


const f = new FormData();
f.append("bg", currentBgUrl);
f.append("fg", fgURL);
f.append("x", X);
f.append("y", Y);
f.append("w", W);
f.append("h", H);
f.append("canvas_w", canvas.width);
f.append("canvas_h", canvas.height);


const out = await safeFetch("/blend", { method: "POST", body: f });


finalImgEl.src = out.final;
finalImgEl.style.display = "block";


} catch (err) {
console.error("generateShadowNow failed", err);
alert("Generate shadow failed → " + err.message);
}
}
window.generateShadowNow = generateShadowNow;

// place3DModelInCanvas: mostly same as original but creates sun control and fabric overlay
function place3DModelInCanvas() {
  if (!lastModelURL) return alert("Generate 3D first!");

  if (fgObj) {
    canvas.remove(fgObj);
    fgObj = null;
  }

  const old = document.getElementById("threeCanvasInFabric");
  if (old) old.remove();

  // html overlay container
  const overlay = document.createElement("div");
  overlay.id = "threeCanvasInFabric";
  overlay.style.position = "absolute";
  overlay.style.left = "200px";
  overlay.style.top = "150px";
  overlay.style.width = "350px";
  overlay.style.height = "350px";
  overlay.style.zIndex = "200";
  overlay.style.pointerEvents = "auto";

  canvasWrapper.appendChild(overlay);

  overlay._baseWidth = overlay.offsetWidth;
  overlay._baseHeight = overlay.offsetHeight;

  const w = overlay.clientWidth;
  const h = overlay.clientHeight;

  // THREE renderer
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true
  });

  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  overlay.appendChild(renderer.domElement);
  window.threeRenderer = renderer;

  // scene
  const scene = new THREE.Scene();
  window.threeScene = scene;

  // camera
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
  camera.position.set(0, 1, 3);
  window.threeCamera = camera;

  // lights
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6));
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));

  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.castShadow = true;
  sun.shadow.mapSize.width = 2048;
  sun.shadow.mapSize.height = 2048;

  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 50;
  sun.shadow.camera.left = -5;
  sun.shadow.camera.right = 5;
  sun.shadow.camera.top = 5;
  sun.shadow.camera.bottom = -5;

  const sunTarget = new THREE.Object3D();
  sunTarget.position.set(0, 0, 0);
  scene.add(sunTarget);
  sun.target = sunTarget;

  sun.position.set(3, 3, 2);
  scene.add(sun);

  window.sunLight = sun;

  // shadow plane
  const shadowPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(6, 6),
    new THREE.ShadowMaterial({ opacity: 0.8 })
  );

  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = -0.6;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  window.shadowPlane = shadowPlane;

  // load model
  const loader = new THREE.OBJLoader();
  loader.load(lastModelURL, obj => {

    let texture = null;

    const applyMaterial = () => {
      obj.traverse(c => {
        if (!c.isMesh) return;

        c.castShadow = true;
        c.receiveShadow = true;

        const mat = new THREE.MeshStandardMaterial({
          color: 0xffffff,
          roughness: 0.6,
          metalness: 0.05,
          map: texture || null
        });

        c.material = mat;
        c.material.needsUpdate = true;
      });
    };

    if (lastTextureURL) {
      new THREE.TextureLoader().load(
        lastTextureURL,
        tex => {
          tex.flipY = false;
          texture = tex;
          applyMaterial();
        },
        undefined,
        () => applyMaterial()
      );
    } else applyMaterial();

    // floor alignment: detect min vertex Y and snap
    const box = new THREE.Box3().setFromObject(obj);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = box.getCenter(new THREE.Vector3());

    // obj.position.sub(center);
    // obj.rotation.z -= 0.17;   // small left/right tilt
    // obj.rotation.x -= 0.16
    obj.updateMatrixWorld(true);

    // compute actual lowest vertex in world space and snap to shadow plane
    let minRealY = Infinity;
    obj.traverse(c => {
      if (!c.isMesh) return;
      const pos = c.geometry.attributes.position;
      const v = new THREE.Vector3();
      for (let i = 0; i < pos.count; i++) {
        v.set(pos.getX(i), pos.getY(i), pos.getZ(i));
        c.localToWorld(v);
        if (v.y < minRealY) minRealY = v.y;
      }
    });

    // snap object's lowest point to the shadow plane Y (automatic)
    const floorY = window.shadowPlane.position.y
    const offset = floorY - minRealY;
    obj.position.y = obj.position.y+offset+0.1;

    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    obj.scale.setScalar(1.5 / maxDim);

    obj.position.x = 0;
    obj.position.z = 0;

    scene.add(obj);
    window.model3D = obj;
  });

  // orbit controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0.5, 0);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // fabric overlay object
  class OverlayObject extends fabric.Group {
    constructor(elem, opt) {
      const rect = new fabric.Rect({
        left: 0,
        top: 0,
        width: elem._baseWidth,
        height: elem._baseHeight,
        fill: "rgba(0,0,0,0)",
        stroke: "black",
        strokeWidth: 1,
        selectable: false,
        evented: false
      });

      super([rect], {
        ...opt,
        hasControls: true,
        hasBorders: true,
        borderColor: "black",
        cornerColor: "white",
        cornerStyle: "rect",
        cornerStrokeColor: "black",
        cornerSize: 14,
        transparentCorners: false,
        rotatingPointOffset: 30,
        padding: 0
      });

      this.html = elem;
      this.rect = rect;

      this.on("selected", () => {
        this.set({
          borderColor: "black",
          cornerColor: "white",
          cornerStrokeColor: "black",
          cornerStyle: "rect",
          transparentCorners: false
        });
      });

      this.on("moving", () => this.sync());
      this.on("scaling", () => this.sync());
      this.on("rotating", () => this.sync());
      this.on("modified", () => this.sync());
    }

    sync() {
      const newW = this.rect.width * this.scaleX;
      const newH = this.rect.height * this.scaleY;

      this.html.style.width = `${newW}px`;
      this.html.style.height = `${newH}px`;

      this.html.style.left = `${this.left}px`;
      this.html.style.top = `${this.top}px`;

      this.html.style.transform = `rotate(${this.angle}deg)`;
      this.html.style.transformOrigin = "top left";

      canvas.renderAll();
    }

    _render() {}
  }

  const fabric3D = new OverlayObject(overlay, {
    left: 200,
    top: 150,
    originX: "left",
    originY: "top"
  });

  window.fabric3D = fabric3D;

  canvas.add(fabric3D);
  canvas.setActiveObject(fabric3D);
  fabric3D.setCoords();
  canvas.renderAll();

  // sun control (draggable circle)
  const sunCtrl = new fabric.Circle({
    radius: 16,
    fill: "yellow",
    stroke: "orange",
    strokeWidth: 3,
    left: fabric3D.left + fabric3D.getScaledWidth() + 40,
    top: fabric3D.top + fabric3D.getScaledHeight() / 2 - 16,
    hasControls: false,
    hasBorders: false,
    lockScalingX: true,
    lockScalingY: true
  });

  canvas.add(sunCtrl);
  window.sunControl = sunCtrl;

  sunCtrl.on("moving", () => {
    updateSunFromSunControl(sunCtrl);
  });

  updateSunFromSunControl(sunCtrl);
}
window.place3DModelInCanvas = place3DModelInCanvas;

// downloadFinal
function downloadFinal() {
  if (!finalImgEl.src) return alert("No final image");

  const a = document.createElement("a");
  a.href = finalImgEl.src;
  a.download = "final.png";
  a.click();
}
window.downloadFinal = downloadFinal;

// multiply3x3
function multiply3x3(A, B) {
  const C = [
    [0,0,0],
    [0,0,0],
    [0,0,0]
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        s += A[i][k] * B[k][j];
      }
      C[i][j] = s;
    }
  }
  return C;
}

// updateShadowSide handler
function updateShadowSide() {
  const sel = document.getElementById("shadowSide");
  if (!sel) return;
  forcedShadowDirection = sel.value || "auto";
}
window.updateShadowSide = updateShadowSide;

// window resize handler
window.addEventListener("resize", () => {
  const overlay = document.getElementById("threeCanvasInFabric");
  if (!overlay || !window.threeRenderer || !window.threeCamera) return;

  const w = overlay.clientWidth;
  const h = overlay.clientHeight;

  window.threeRenderer.setSize(w, h);
  window.threeCamera.aspect = w / h;
  window.threeCamera.updateProjectionMatrix();
});

// debugDrawFeet (unchanged)
function debugDrawFeet() {
  const overlay = document.getElementById("threeCanvasInFabric");
  if (!overlay) return;

  let sv = document.getElementById("feetDebugSvg");
  if (!sv) {
    sv = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    sv.id = "feetDebugSvg";

    sv.style.position = "absolute";
    sv.style.left = "0";
    sv.style.top = "0";
    sv.style.width = overlay.clientWidth + "px";
    sv.style.height = overlay.clientHeight + "px";
    sv.style.pointerEvents = "none";

    overlay.appendChild(sv);
  }

  sv.innerHTML = "";
  const feet = getModelFootPointsInOverlay();

  for (let p of feet) {
    const c = document.createElementNS("http://www.w3.org/2000/svg","circle");
    c.setAttribute("cx", p.x);
    c.setAttribute("cy", p.y);
    c.setAttribute("r", 6);
    c.setAttribute("fill", "red");
    c.setAttribute("stroke", "white");
    sv.appendChild(c);
  }
}
window.debugDrawFeet = debugDrawFeet;