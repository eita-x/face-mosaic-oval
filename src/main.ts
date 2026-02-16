import JSZip from "jszip";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const fileEl = document.getElementById("file") as HTMLInputElement;
const preview = document.getElementById("preview") as HTMLCanvasElement;
const pctx = preview.getContext("2d")!;
const strengthEl = document.getElementById("strength") as HTMLInputElement;
const strengthValEl = document.getElementById("strengthVal") as HTMLSpanElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const downloadZipBtn = document.getElementById("downloadZip") as HTMLButtonElement;
const dropEl = document.getElementById("drop") as HTMLDivElement;

strengthEl.addEventListener("input", () => {
  strengthValEl.textContent = strengthEl.value;
});

let landmarker: FaceLandmarker | null = null;
let lastZipBlob: Blob | null = null;

async function initLandmarker() {
  if (landmarker) return landmarker;

  statusEl.textContent = "初期化中…";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  const modelAssetPath =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath,
      // 写真のみ＆安定優先：CPUでOK
      // delegate: "GPU",
    },
    runningMode: "IMAGE",
    // 複数"人"も一応多めに
    numFaces: 20,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  statusEl.textContent = "準備OK";
  return landmarker;
}

/**
 * Face Oval（輪郭）ループ（MediaPipe Face Meshでよく使われる）
 */
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
  361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109,
];

type Landmark = { x: number; y: number; z?: number };

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function buildMosaicPatch(
  sourceCanvas: HTMLCanvasElement,
  x: number,
  y: number,
  w: number,
  h: number,
  blockSize: number
) {
  const tmp1 = document.createElement("canvas");
  tmp1.width = Math.ceil(w);
  tmp1.height = Math.ceil(h);
  const t1 = tmp1.getContext("2d")!;
  t1.drawImage(sourceCanvas, x, y, w, h, 0, 0, tmp1.width, tmp1.height);

  const sw = Math.max(1, Math.floor(tmp1.width / blockSize));
  const sh = Math.max(1, Math.floor(tmp1.height / blockSize));

  const tmp2 = document.createElement("canvas");
  tmp2.width = sw;
  tmp2.height = sh;
  const t2 = tmp2.getContext("2d")!;
  t2.imageSmoothingEnabled = false;
  t2.drawImage(tmp1, 0, 0, sw, sh);

  const out = document.createElement("canvas");
  out.width = tmp1.width;
  out.height = tmp1.height;
  const o = out.getContext("2d")!;
  o.imageSmoothingEnabled = false;
  o.drawImage(tmp2, 0, 0, sw, sh, 0, 0, out.width, out.height);

  return out;
}

function mosaicFaceOval(
  ctx2d: CanvasRenderingContext2D,
  landmarks: Landmark[],
  imgW: number,
  imgH: number,
  blockSize: number
) {
  const pts = FACE_OVAL
    .map((i) => landmarks[i])
    .filter(Boolean)
    .map((p) => ({ x: p.x * imgW, y: p.y * imgH }));

  if (pts.length < 3) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  // 余白（漏れ対策。必要なら 0.10〜0.14 に上げてOK）
  const pad = 0.10;
  const w = maxX - minX;
  const h = maxY - minY;

  const x = clamp(minX - w * pad, 0, imgW);
  const y = clamp(minY - h * pad, 0, imgH);
  const cw = clamp(w * (1 + pad * 2), 1, imgW - x);
  const ch = clamp(h * (1 + pad * 2), 1, imgH - y);

  const mosaicked = buildMosaicPatch(ctx2d.canvas, x, y, cw, ch, blockSize);

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

async function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return await new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png");
  });
}

function safeBaseName(name: string) {
  // 拡張子を外して安全なファイル名へ
  const base = name.replace(/\.[^.]+$/, "");
  return base.replace(/[\\/:*?"<>|]/g, "_").trim() || "image";
}

async function processOneFile(file: File, index: number, total: number): Promise<{ name: string; blob: Blob }> {
  const lm = await initLandmarker();
  const strength = Number(strengthEl.value);

  statusEl.textContent = `処理中… ${index}/${total}：${file.name}`;

  const img = await createImageBitmap(file);

  // オフスクリーン（DOMに出さない）で処理
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(img, 0, 0);

  const res = lm.detect(img);
  const faces = res.faceLandmarks ?? [];

  // 1枚プレビュー（最後に処理したやつを表示）
  preview.width = img.width;
  preview.height = img.height;
  pctx.imageSmoothingEnabled = true;
  pctx.clearRect(0, 0, preview.width, preview.height);
  pctx.drawImage(img, 0, 0);

  if (faces.length === 0) {
    // 顔なしでも"そのまま"返す（運用的に便利）
    statusEl.textContent = `顔なし：${file.name}（そのまま保存）`;
    const blob = await canvasToPngBlob(c);
    return { name: `${safeBaseName(file.name)}_mosaic.png`, blob };
  }

  for (const face of faces) {
    mosaicFaceOval(ctx, face as Landmark[], img.width, img.height, strength);
    // previewにも同じモザイク（見た目確認用）
    mosaicFaceOval(pctx, face as Landmark[], img.width, img.height, strength);
  }

  const blob = await canvasToPngBlob(c);
  return { name: `${safeBaseName(file.name)}_mosaic.png`, blob };
}

async function handleFiles(files: File[]) {
  if (files.length === 0) return;

  // 画像だけに絞る
  const imgs = files.filter((f) => f.type.startsWith("image/"));
  if (imgs.length === 0) {
    statusEl.textContent = "画像ファイルが見つからない";
    return;
  }

  downloadZipBtn.disabled = true;
  lastZipBlob = null;

  const zip = new JSZip();

  try {
    // 逐次処理（メモリ/CPU食い過ぎ防止）
    let i = 0;
    for (const f of imgs) {
      i += 1;
      const out = await processOneFile(f, i, imgs.length);
      zip.file(out.name, out.blob);
    }

    statusEl.textContent = `ZIP生成中…（${imgs.length}枚）`;
    lastZipBlob = await zip.generateAsync({ type: "blob" });

    downloadZipBtn.disabled = false;
    statusEl.textContent = `完了：${imgs.length}枚（ZIP保存OK）`;
  } catch (e) {
    console.error(e);
    statusEl.textContent = "エラー（コンソール見て）";
  }
}

fileEl.addEventListener("change", async () => {
  await handleFiles(Array.from(fileEl.files ?? []));
});

// ドラッグ＆ドロップ
dropEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropEl.style.background = "#f0f7ff";
  dropEl.style.borderColor = "#5aa6ff";
});

dropEl.addEventListener("dragleave", () => {
  dropEl.style.background = "#fafafa";
  dropEl.style.borderColor = "#bbb";
});

dropEl.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropEl.style.background = "#fafafa";
  dropEl.style.borderColor = "#bbb";

  const dt = e.dataTransfer;
  if (!dt) return;

  const files = Array.from(dt.files ?? []);
  await handleFiles(files);
});

downloadZipBtn.addEventListener("click", () => {
  if (!lastZipBlob) return;

  const url = URL.createObjectURL(lastZipBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "face-oval-mosaic.zip";
  a.click();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// 初期表示
statusEl.textContent = "画像を選ぶか、ドラッグ＆ドロップしてね";
