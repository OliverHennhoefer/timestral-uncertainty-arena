/** Minimal reader for the safetensors format (https://github.com/huggingface/safetensors). */

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Parse a safetensors buffer into float32 tensors (F32, F16 and BF16 are supported). */
export function parseSafetensors(buffer: ArrayBuffer): Map<string, Tensor> {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerText = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
  const header = JSON.parse(headerText) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const base = 8 + headerLen;
  const out = new Map<string, Tensor>();
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [begin, end] = info.data_offsets;
    const start = base + begin;
    const nbytes = end - begin;
    let data: Float32Array;
    if (info.dtype === "F32") {
      data = start % 4 === 0 ? new Float32Array(buffer, start, nbytes / 4) : new Float32Array(buffer.slice(start, start + nbytes));
    } else if (info.dtype === "F16" || info.dtype === "BF16") {
      const n = nbytes / 2;
      data = new Float32Array(n);
      const dv = new DataView(buffer, start, nbytes);
      const f32 = new Float32Array(1);
      const u32 = new Uint32Array(f32.buffer);
      for (let i = 0; i < n; i++) {
        const h = dv.getUint16(i * 2, true);
        if (info.dtype === "BF16") {
          u32[0] = h << 16;
          data[i] = f32[0];
        } else {
          data[i] = f16ToF32(h);
        }
      }
    } else {
      continue; // integer buffers are not used at inference
    }
    out.set(name, { shape: info.shape, data });
  }
  return out;
}

/** Serialize float32 tensors to safetensors (used by tests). */
export function writeSafetensors(tensors: Record<string, Tensor>): ArrayBuffer {
  const header: Record<string, unknown> = {};
  let offset = 0;
  const entries = Object.entries(tensors);
  for (const [name, t] of entries) {
    header[name] = { dtype: "F32", shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
    offset += t.data.byteLength;
  }
  let headerText = JSON.stringify(header);
  while ((8 + headerText.length) % 8 !== 0) headerText += " ";
  const headerBytes = new TextEncoder().encode(headerText);
  const buf = new ArrayBuffer(8 + headerBytes.length + offset);
  new DataView(buf).setBigUint64(0, BigInt(headerBytes.length), true);
  new Uint8Array(buf, 8, headerBytes.length).set(headerBytes);
  let pos = 8 + headerBytes.length;
  for (const [, t] of entries) {
    new Uint8Array(buf, pos, t.data.byteLength).set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
    pos += t.data.byteLength;
  }
  return buf;
}
