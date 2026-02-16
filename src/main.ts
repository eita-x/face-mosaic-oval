import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const fileEl = document.getElementById("file") as HTMLInputElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const strengthEl = document.getElementById("strength") as HTMLInputElement;
const strengthValEl = document.getElementById("strengthVal") as HTMLSpanElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const downloadBtn = document.getElementById("download") as HTMLButtonElement;

strengthEl.addEventListener("input", () => {
  strengthValEl.textContent = strengthEl.value;
});

let landmarker: FaceLandmarker | null = null;

async function initLandmarker() {
  if (landmarker) return landmarker;

  statusEl.textContent = "初期化中…";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // 公式配布のモデルURL例（taskファイル）
  const modelAssetPath =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath,
      // delegate: "GPU", // 写真だけならCPUで十分。GPUは環境差でハマりやすい
    },
    runningMode: "IMAGE",
    numFaces: 10,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  statusEl.textContent = "準備OK";
  return landmarker;
}

/**
 * Face Oval（輪郭）に使う代表的な点インデックス
 * ※MediaPipe Face Meshでよく使われる輪郭ループ
 */
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
  361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109,
];

async function processFile(f: File) {
  try {
    const lm = await initLandmarker();
    statusEl.textContent = "画像読み込み中…";

    const img = await createImageBitmap(f);

    // 実ピクセルで描画
    canvas.width = img.width;
    canvas.height = img.height;

    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    statusEl.textContent = "輪郭検出中…";
    const res = lm.detect(img);
    const faces = res.faceLandmarks ?? [];

    if (faces.length === 0) {
      statusEl.textContent = "顔が見つからなかった";
      return;
    }

    // 1顔ずつ輪郭でクリップしてモザイク
    for (const face of faces) {
      mosaicFaceOval(ctx, face, img.width, img.height);
    }

    statusEl.textContent = `完了：${faces.length}人`;
  } catch (e) {
    console.error(e);
    statusEl.textContent = "エラー（コンソール見て）";
  }
}

fileEl.addEventListener("change", () => {
  const f = fileEl.files?.[0];
  if (f) processFile(f);
});

// ドラッグ＆ドロップ対応
document.addEventListener("dragover", (e) => {
  e.preventDefault();
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files[0];
  if (f && f.type.startsWith("image/")) processFile(f);
});

downloadBtn.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "face-oval-mosaic.png";
  a.href = canvas.toDataURL("image/png");
  a.click();
});

type Landmark = { x: number; y: number; z?: number };

function mosaicFaceOval(
  ctx2d: CanvasRenderingContext2D,
  landmarks: Landmark[],
  imgW: number,
  imgH: number,
) {
  // 輪郭点をピクセル座標に
  const rawPts = FACE_OVAL
    .map((i) => landmarks[i])
    .filter(Boolean)
    .map((p) => ({ x: p.x * imgW, y: p.y * imgH }));

  if (rawPts.length < 3) return;

  // 重心を求めて12%拡大
  const cx = rawPts.reduce((s, p) => s + p.x, 0) / rawPts.length;
  const cy = rawPts.reduce((s, p) => s + p.y, 0) / rawPts.length;
  const expand = 1.12;
  const pts = rawPts.map((p) => ({
    x: cx + (p.x - cx) * expand,
    y: cy + (p.y - cy) * expand,
  }));

  // 輪郭の外接バウンディング（モザイク処理の範囲を最小化＝軽い）
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  const w = maxX - minX;
  const h = maxY - minY;

  // 顔幅に応じてブロックサイズを自動調整（スライダー値は無視）
  const effectiveBlock = Math.max(12, Math.min(32, w / 12));

  // 余白（輪郭のズレ対策、少しだけ）
  const pad = 0.08;

  const x = clamp(minX - w * pad, 0, imgW);
  const y = clamp(minY - h * pad, 0, imgH);
  const cw = clamp(w * (1 + pad * 2), 1, imgW - x);
  const ch = clamp(h * (1 + pad * 2), 1, imgH - y);

  // ① まず「範囲だけ」モザイク画像を作る（小さく→拡大）
  const mosaicked = buildMosaicPatch(ctx2d.canvas, x, y, cw, ch, effectiveBlock);

  // ② 輪郭でクリップして、その中だけ貼る（輪郭ピッタリ）
  ctx2d.save();
  ctx2d.beginPath();
  ctx2d.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx2d.lineTo(pts[i].x, pts[i].y);
  ctx2d.closePath();
  ctx2d.clip();

  ctx2d.imageSmoothingEnabled = false;
  ctx2d.drawImage(mosaicked, x, y);

  ctx2d.restore();
}

function buildMosaicPatch(
  sourceCanvas: HTMLCanvasElement,
  x: number,
  y: number,
  w: number,
  h: number,
  blockSize: number
) {
  // tmp1: 切り出し
  const tmp1 = document.createElement("canvas");
  tmp1.width = Math.ceil(w);
  tmp1.height = Math.ceil(h);
  const t1 = tmp1.getContext("2d")!;
  t1.drawImage(sourceCanvas, x, y, w, h, 0, 0, tmp1.width, tmp1.height);

  // tmp2: 縮小
  const sw = Math.max(1, Math.floor(tmp1.width / blockSize));
  const sh = Math.max(1, Math.floor(tmp1.height / blockSize));
  const tmp2 = document.createElement("canvas");
  tmp2.width = sw;
  tmp2.height = sh;
  const t2 = tmp2.getContext("2d")!;
  t2.imageSmoothingEnabled = false;
  t2.drawImage(tmp1, 0, 0, sw, sh);

  // out: 拡大（nearest）
  const out = document.createElement("canvas");
  out.width = tmp1.width;
  out.height = tmp1.height;
  const o = out.getContext("2d")!;
  o.imageSmoothingEnabled = false;
  o.drawImage(tmp2, 0, 0, sw, sh, 0, 0, out.width, out.height);

  return out;
}

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}
