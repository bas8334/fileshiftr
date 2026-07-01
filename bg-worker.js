/* ============================================================================
   bg-worker.js — Achtergrondverwijdering in de browser (Fileshiftr)
   ----------------------------------------------------------------------------
   Draait het AI-model in een Web Worker, zodat de UI soepel blijft.
   Model: BiRefNet_lite (MIT-licentie) via de onnx-community ONNX-conversie.
          Lichter dan het volledige BiRefNet, zodat het in de browser past.
   Bibliotheek: Transformers.js v3 (vastgepind op een exacte versie).

   GEBRUIK:
   - Fase 1 (meteen live): laat SELF_HOST op false. Het model komt van de
     publieke CDN. Geen Cloudflare R2 nodig. Je foto blijft op je apparaat.
   - Fase 2 (zelf hosten): zet SELF_HOST op true en MODEL_HOST naar jouw
     R2-domein. De map-structuur op R2 moet dan exact zijn:
       https://MODEL_HOST/onnx-community/BiRefNet_lite-ONNX/config.json
       https://MODEL_HOST/onnx-community/BiRefNet_lite-ONNX/preprocessor_config.json
       https://MODEL_HOST/onnx-community/BiRefNet_lite-ONNX/onnx/model.onnx

   Geen COOP/COEP-headers nodig: WASM draait single-thread en WebGPU heeft
   die headers niet nodig.
   ============================================================================ */

import {
  env, AutoModel, AutoProcessor, RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

/* ============================================================================
   SCHAKELAAR — kies waar het MODELBESTAND vandaan komt.
   (Je foto blijft in BEIDE gevallen 100% op het apparaat; dit gaat alleen
    over waar het programmabestand 'model.onnx' wordt opgehaald.)

   false = Fase 1: model van de publieke CDN. Geen R2 nodig. Meteen live.
   true  = Fase 2: model van JOUW eigen R2-host (zet hieronder MODEL_HOST).
   ============================================================================ */
const SELF_HOST  = false;
const MODEL_HOST = "https://models.fileshiftr.com/";   // gebruikt als SELF_HOST = true

if (SELF_HOST) {
  env.allowRemoteModels = false;            // nooit naar de publieke CDN
  env.allowLocalModels  = true;
  env.localModelPath    = MODEL_HOST;       // modelbestanden van jouw R2
} else {
  env.allowRemoteModels = true;             // modelbestanden van de publieke CDN
  env.allowLocalModels  = false;
}
// De WASM-runtime laat ik standaard van de CDN komen — die bevat geen
// gebruikersdata, dus dat raakt de privacybelofte niet. (Zie notitie onderaan
// als je later óók de WASM-bestanden zelf wilt hosten.)
env.backends.onnx.wasm.numThreads = 1;      // single-thread -> geen COOP/COEP nodig

const MODEL_ID = "onnx-community/BiRefNet_lite-ONNX";   // lichter (~200 MB), draait in de browser

/* WebGPU heeft een bekende bug in onnxruntime-web met de Slice-operatie van
   BiRefNet ("Invalid ComputePipeline computeSliceOffsets"), dus we draaien op
   WASM — de configuratie die de modelmakers zelf als werkend documenteren.
   Werkt WASM goed en wil je later experimenteren met WebGPU? Zet dit op true. */
const TRY_WEBGPU = false;

let model = null;
let processor = null;
let currentDevice = null;

async function hasWebGPU() {
  try {
    if (self.navigator && self.navigator.gpu) {
      const adapter = await self.navigator.gpu.requestAdapter();
      if (adapter) return true;
    }
  } catch (_e) { /* nee */ }
  return false;
}

async function loadOn(device) {
  // fp32 voor beide routes: het lite-model is klein genoeg, en zo hergebruikt
  // een WebGPU->WASM-fallback hetzelfde (al gedownloade) bestand — geen 2e download.
  const dtype = "fp32";
  const progress_callback = (p) => self.postMessage({ type: "progress", data: p });
  model = await AutoModel.from_pretrained(MODEL_ID, { device, dtype, progress_callback });
  if (!processor) processor = await AutoProcessor.from_pretrained(MODEL_ID);
  currentDevice = device;
}

/* Het modeloutput-tensor robuust ophalen (sleutelnaam kan verschillen) */
function pickOutput(out) {
  if (out && out.output_image) return out.output_image;
  if (out && out.logits) return out.logits;
  const vals = Object.values(out || {});
  return vals.length ? vals[0] : null;
}

/* Maak van wat dan ook (Error, string, getal, abort) een leesbare tekst */
function describe(err) {
  if (err === null || err === undefined) return "no error object";
  if (typeof err === "string") return err;
  if (typeof err === "number") return "abort code " + err;
  const parts = [];
  if (err.name) parts.push(err.name);
  if (err.message) parts.push(err.message);
  if (parts.length === 0) {
    try { parts.push(JSON.stringify(err)); } catch (_e) { parts.push(String(err)); }
  }
  return parts.join(": ");
}

/* Eén inferentie -> 1-kanaals masker (Uint8) op modelresolutie.
   Houdt bij in welke STAP het misgaat, zodat fouten herkenbaar zijn. */
async function infer(image) {
  let stage = "start";
  try {
    stage = "processor";
    const inputs = await processor(image);
    const pixel_values = inputs.pixel_values;

    stage = "model";
    const out = await model({ input_image: pixel_values });

    stage = "output";
    const tensor = pickOutput(out);
    if (!tensor) throw new Error("model gaf geen output-tensor (sleutels: " + Object.keys(out || {}).join(",") + ")");

    stage = "mask";
    const maskImg = await RawImage.fromTensor(
      tensor[0].sigmoid().mul(255).to("uint8")
    );
    const w = maskImg.width, h = maskImg.height;
    const ch = maskImg.data.length / (w * h);
    let mask;
    if (ch > 1) {
      mask = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) mask[i] = maskImg.data[i * ch];
    } else {
      mask = new Uint8Array(maskImg.data);
    }
    return { mask, width: w, height: h };
  } catch (err) {
    try { console.error("bg-worker infer error @" + stage, err); } catch (_e) {}
    throw new Error("@" + stage + ": " + describe(err));
  }
}

self.onmessage = async (e) => {
  const msg = e.data || {};

  if (msg.type === "load") {
    try {
      const device = (TRY_WEBGPU && await hasWebGPU()) ? "webgpu" : "wasm";
      try {
        await loadOn(device);
      } catch (err) {
        if (device === "webgpu") { await loadOn("wasm"); }   // laden mislukt -> wasm
        else throw err;
      }
      self.postMessage({ type: "ready", device: currentDevice });
    } catch (err) {
      try { console.error("bg-worker load error", err); } catch (_e) {}
      self.postMessage({ type: "error", message: "load: " + describe(err) });
    }
    return;
  }

  if (msg.type === "run") {
    try {
      if (!model || !processor) throw new Error("model not loaded");

      // 1) foto decoderen (komt binnen als databuffer)
      let image;
      try {
        const blob = new Blob([msg.buf], { type: msg.mime || "image/png" });
        image = await RawImage.fromBlob(blob);
      } catch (errDec) {
        try { console.error("bg-worker decode error", errDec); } catch (_e) {}
        throw new Error("@decode: " + describe(errDec));
      }

      // 2) inferentie, met WebGPU->WASM fallback
      let res;
      try {
        res = await infer(image);
      } catch (errGpu) {
        if (currentDevice === "webgpu") {
          self.postMessage({ type: "status", message: "Switching to compatibility mode…" });
          try {
            await loadOn("wasm");
            self.postMessage({ type: "ready", device: "wasm" });
            res = await infer(image);
          } catch (errWasm) {
            throw new Error("webgpu " + describe(errGpu) + "  ||  wasm " + describe(errWasm));
          }
        } else {
          throw errGpu;
        }
      }

      self.postMessage(
        { type: "result", mask: res.mask, width: res.width, height: res.height },
        [res.mask.buffer]
      );
    } catch (err) {
      try { console.error("bg-worker run error", err); } catch (_e) {}
      self.postMessage({ type: "error", message: "[" + (currentDevice || "?") + "] " + describe(err) });
    }
    return;
  }
};

/* ----------------------------------------------------------------------------
   OPTIONEEL LATER
   1) Ook de WASM-runtime zelf hosten (volledige onafhankelijkheid van de CDN):
      kopieer de ort-wasm-bestanden naar /wasm/ op je R2 en zet
        env.backends.onnx.wasm.wasmPaths = MODEL_HOST + "wasm/";
   2) Sneller op de WASM-route: multithread aanzetten
        env.backends.onnx.wasm.numThreads = 4;
      Dat vereist cross-origin isolation (COOP/COEP-headers via _headers op
      Pages, plus 'Cross-Origin-Resource-Policy: cross-origin' op je R2). COEP
      kan andere externe embeds (ads/analytics) breken, dus alleen doen als de
      WASM-route te traag blijkt. WebGPU heeft dit allemaal NIET nodig.
   ---------------------------------------------------------------------------- */
