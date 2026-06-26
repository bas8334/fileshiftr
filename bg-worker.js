/* ============================================================================
   bg-worker.js — Achtergrondverwijdering in de browser (Fileshiftr)
   ----------------------------------------------------------------------------
   Draait het AI-model in een Web Worker, zodat de UI soepel blijft.
   Model: BiRefNet (MIT-licentie) via de onnx-community ONNX-conversie.
   Bibliotheek: Transformers.js v3 (vastgepind op een exacte versie).

   GEBRUIK:
   - Fase 1 (meteen live): laat SELF_HOST op false. Het model komt van de
     publieke CDN. Geen Cloudflare R2 nodig. Je foto blijft op je apparaat.
   - Fase 2 (zelf hosten): zet SELF_HOST op true en MODEL_HOST naar jouw
     R2-domein. De map-structuur op R2 moet dan exact zijn:
       https://MODEL_HOST/onnx-community/BiRefNet-ONNX/config.json
       https://MODEL_HOST/onnx-community/BiRefNet-ONNX/preprocessor_config.json
       https://MODEL_HOST/onnx-community/BiRefNet-ONNX/onnx/model_fp16.onnx
       https://MODEL_HOST/onnx-community/BiRefNet-ONNX/onnx/model.onnx

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

const MODEL_ID = "onnx-community/BiRefNet-ONNX";

let model = null;
let processor = null;

/* WebGPU detecteren; anders WASM */
async function pickDevice() {
  try {
    if (self.navigator && self.navigator.gpu) {
      const adapter = await self.navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    }
  } catch (_e) { /* val terug op wasm */ }
  return "wasm";
}

async function loadModel() {
  let device = await pickDevice();
  let dtype  = device === "webgpu" ? "fp16" : "fp32";   // alleen deze bestaan in deze repo

  const progress_callback = (p) => self.postMessage({ type: "progress", data: p });

  try {
    model = await AutoModel.from_pretrained(MODEL_ID, { device, dtype, progress_callback });
  } catch (err) {
    // WebGPU-initialisatie mislukt -> terugvallen op WASM (fp32)
    device = "wasm"; dtype = "fp32";
    model = await AutoModel.from_pretrained(MODEL_ID, { device, dtype, progress_callback });
  }
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  return device;
}

/* Het modeloutput-tensor robuust ophalen (sleutelnaam kan verschillen) */
function pickOutput(out) {
  if (out && out.output_image) return out.output_image;
  if (out && out.logits) return out.logits;
  const vals = Object.values(out || {});
  return vals.length ? vals[0] : null;
}

self.onmessage = async (e) => {
  const msg = e.data || {};

  if (msg.type === "load") {
    try {
      const device = await loadModel();
      self.postMessage({ type: "ready", device });
    } catch (err) {
      self.postMessage({ type: "error", message: (err && err.message) || "model load failed" });
    }
    return;
  }

  if (msg.type === "run") {
    try {
      if (!model || !processor) throw new Error("model not loaded");

      const image = await RawImage.fromURL(msg.url);
      const { pixel_values } = await processor(image);
      const out = await model({ input_image: pixel_values });

      const tensor = pickOutput(out);
      if (!tensor) throw new Error("no model output");

      // logits -> 0..255 alpha-masker, op model-resolutie
      const maskImg = await RawImage.fromTensor(
        tensor[0].sigmoid().mul(255).to("uint8")
      );

      // forceer 1 kanaal (veiligheidsnet) en stuur als transferable Uint8
      const w = maskImg.width, h = maskImg.height;
      const ch = maskImg.data.length / (w * h);     // verwacht 1
      let mask = maskImg.data;
      if (ch > 1) {
        const single = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) single[i] = maskImg.data[i * ch];
        mask = single;
      } else {
        mask = new Uint8Array(maskImg.data);         // kopie t.b.v. transfer
      }
      self.postMessage(
        { type: "result", mask, width: w, height: h },
        [mask.buffer]
      );
    } catch (err) {
      self.postMessage({ type: "error", message: (err && err.message) || "run failed" });
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
