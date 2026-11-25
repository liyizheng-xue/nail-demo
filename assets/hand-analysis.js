// 手型/肤色分析：全部在浏览器里跑，照片不离开设备。
// 只对外给「人话结论」，不给 L*/a*/b*/ITA 这些实验室数字。
import { FilesetResolver, HandLandmarker } from './mp/vision_bundle.mjs';

let landmarker = null;

async function ensureModel(){
  if (landmarker) return landmarker;
  const vision = await FilesetResolver.forVisionTasks(new URL('./mp/', import.meta.url).href);
  const opts = {
    baseOptions: {
      modelAssetPath: new URL('./mp/hand_landmarker.task', import.meta.url).href,
      delegate: 'GPU'
    },
    runningMode: 'IMAGE',
    numHands: 1,
    minHandDetectionConfidence: 0.4,
    minHandPresenceConfidence: 0.4,
    minTrackingConfidence: 0.4
  };
  try {
    landmarker = await HandLandmarker.createFromOptions(vision, opts);
  } catch (e) {
    opts.baseOptions.delegate = 'CPU';
    landmarker = await HandLandmarker.createFromOptions(vision, opts);
  }
  return landmarker;
}

function dataUrlToBlob(u){
  const i = u.indexOf(',');
  const head = u.slice(0, i), body = u.slice(i + 1);
  const mime = (/data:(.*?);/.exec(head) || [])[1] || 'image/jpeg';
  const bin = atob(body);
  const arr = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) arr[k] = bin.charCodeAt(k);
  return new Blob([arr], { type: mime });
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function bgRef(img){
  const { data, width: W, height: H } = img;
  const pts = [[7, 7], [W - 8, 7], [7, H - 8], [W - 8, H - 8]];
  const rs = [], gs = [], bs = [];
  pts.forEach(([x, y]) => {
    for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++){
      const i = ((y + dy) * W + (x + dx)) * 4;
      rs.push(data[i]); gs.push(data[i + 1]); bs.push(data[i + 2]);
    }
  });
  const med = a => a.sort((p, q) => p - q)[Math.floor(a.length / 2)] || 0;
  return [med(rs), med(gs), med(bs)];
}

function scanWidth(img, a, b, skin, bg, at){
  const { data, width: W, height: H } = img;
  const px = a.x + (b.x - a.x) * at, py = a.y + (b.y - a.y) * at;
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len;
  const maxOff = Math.max(8, Math.min(80, len * 1.4));
  const px3 = t => {
    const x = Math.round(px + nx * t), y = Math.round(py + ny * t);
    if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) return null;
    const i = (y * W + x) * 4;
    return [data[i], data[i + 1], data[i + 2]];
  };
  const isBody = t => {
    const c = px3(t); if (!c) return false;
    const dBg = Math.hypot(c[0] - bg[0], c[1] - bg[1], c[2] - bg[2]);
    const dSk = Math.hypot(c[0] - skin[0], c[1] - skin[1], c[2] - skin[2]);
    return dBg > 55 || dSk < 40;
  };
  let l = 0, r = 0, gap = 0;
  for (let t = 1; t <= maxOff; t++){ if (isBody(t)) { l = t; gap = 0; } else { if (++gap > 2) break; } }
  gap = 0;
  for (let t = 1; t <= maxOff; t++){ if (isBody(-t)) { r = t; gap = 0; } else { if (++gap > 2) break; } }
  return l + r;
}

function medianPatch(img, cx, cy, r){
  const { data, width: W, height: H } = img;
  const rs = [], gs = [], bs = [];
  for (let y = cy - r; y <= cy + r; y += 2){
    for (let x = cx - r; x <= cx + r; x += 2){
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      const i = (y * W + x) * 4;
      rs.push(data[i]); gs.push(data[i + 1]); bs.push(data[i + 2]);
    }
  }
  const med = arr => arr.sort((p, q) => p - q)[Math.floor(arr.length / 2)] || 0;
  return [med(rs), med(gs), med(bs)];
}

function rgb2lab(rgb){
  const f = v => { v /= 255; return v > 0.04045 ? Math.pow((v + 0.055) / 1.055, 2.4) : v / 12.92; };
  const r = f(rgb[0]), g = f(rgb[1]), b = f(rgb[2]);
  let X = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
  let Y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
  let Z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
  const t = v => v > 0.008856 ? Math.cbrt(v) : (7.787 * v + 16 / 116);
  X = t(X); Y = t(Y); Z = t(Z);
  return { L: 116 * Y - 16, a: 500 * (X - Y), b: 200 * (Y - Z) };
}

async function toBlob(src){
  if (src.startsWith('data:')) return dataUrlToBlob(src);
  const r = await fetch(src);
  if (!r.ok) throw new Error('图片取不到');
  return await r.blob();
}

// 返回：人话标签 + 少量给后端挑款用的数值
export async function analyzeImage(src){
  const bmp = await createImageBitmap(await toBlob(src));
  const lm = await ensureModel();
  const res = lm.detect(bmp);
  if (!res || !res.landmarks || !res.landmarks.length) return { ok: false, reason: 'no-hand' };
  const raw = res.landmarks[0];
  const P = i => ({ x: raw[i].x * bmp.width, y: raw[i].y * bmp.height });

  const cap = 1000;
  const s2 = Math.min(1, cap / Math.max(bmp.width, bmp.height));
  const ow = Math.round(bmp.width * s2), oh = Math.round(bmp.height * s2);
  const oc = document.createElement('canvas'); oc.width = ow; oc.height = oh;
  const octx = oc.getContext('2d'); octx.drawImage(bmp, 0, 0, ow, oh);
  const img = octx.getImageData(0, 0, ow, oh);
  const Q = i => ({ x: raw[i].x * ow, y: raw[i].y * oh });

  const core = [0, 5, 9, 13, 17].map(Q);
  const cx = core.reduce((s, p) => s + p.x, 0) / 5, cy = core.reduce((s, p) => s + p.y, 0) / 5;
  const skin = medianPatch(img, Math.round(cx), Math.round(cy), 9);

  const palmLen = dist(P(0), P(9));
  const palmWid = dist(P(5), P(17));
  const handRatio = palmLen / palmWid;
  const d24 = dist(P(5), P(8)) / dist(P(13), P(16));

  const bg = bgRef(img);
  const fingers = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
  const tips = fingers.map(f => {
    const a = Q(f[2]), b = Q(f[3]);
    const len = dist(a, b);
    const ws = [0.45, 0.6, 0.75].map(k => scanWidth(img, a, b, skin, bg, k)).sort((p, q) => p - q);
    const wid = ws[1];
    return { len, wid, ratio: wid > 3 ? len / wid : 0 };
  }).filter(t => t.ratio > 0);
  const tipRatio = tips.length ? tips.reduce((s, t) => s + t.ratio, 0) / tips.length : 0;

  const lab = rgb2lab(skin);
  const ita = Math.atan((lab.L - 50) / Math.max(1e-3, lab.b)) * 180 / Math.PI;
  const toneLabel = lab.a >= 0.75 * lab.b ? '偏粉（冷调倾向）'
                  : (lab.a >= 0.55 * lab.b ? '中性' : '偏黄（暖调）');
  const lightLabel = ita >= 55 ? '很浅' : ita >= 41 ? '偏浅' : ita >= 28 ? '中等' : ita >= 10 ? '偏深' : '较深';

  return {
    ok: true,
    handRatio, tipRatio, d24, lab, ita,
    toneLabel, lightLabel,
    handLabel: handRatio >= 1.40 ? '偏修长' : handRatio >= 1.20 ? '均衡' : '偏宽厚',
    tipLabel: tipRatio >= 1.60 ? '偏修长' : tipRatio >= 1.25 ? '适中' : '偏短宽',
    needWhite: (lab.b >= 16 || lab.L < 62) ? 4 : 3,
    at: Date.now()
  };
}
