// Upstream wire translators: every provider wire (openai, anthropic, gemini) normalized to an
// internal OpenAI-shaped answer, so the client side only ever speaks two wires.
//
// The trial's providers speak their real service wires:
//   - openai:    POST /v1/chat/completions            (JSON or SSE chat.completion(.chunk))
//   - anthropic: POST /v1/messages                    (JSON or SSE message events)
//   - gemini:    POST /v1beta/models/<m>:generateContent | :streamGenerateContent?alt=sse
//
// This file converts a client request (OpenAI Chat Completions shape — main.ts normalizes the
// Anthropic client body into that shape first) into an upstream request on the provider's wire,
// and converts the upstream answer back into an internal Chat Completions answer (JSON or a
// chunk stream). Everything is a pure function of its arguments: no clock, no randomness.
//
// Usage passthrough: the scorer checks the client's usage equals the provider's usage, so the
// normalized answer carries the upstream usage untouched (prompt_tokens/completion_tokens).

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopFromFinish(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishFromStop(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function finishFromGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'PROHIBITED_CONTENT' || reason === 'BLOCKED') return 'content_filter';
  if (reason === 'MALFORMED_FUNCTION_CALL') return 'tool_calls';
  return 'stop';
}

/** A tool call's arguments as an object: the upstream's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** The text of a string content or of an array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

// ---------------------------------------------------------------------------
// Anthropic upstream: an internal Chat Completions request as a Messages request.
// ---------------------------------------------------------------------------

/** An internal OpenAI-shaped request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    if (raw.role === 'system') {
      // The system message goes to the request's system field, joined with any earlier one.
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    // A tool message becomes a user turn carrying a tool_result block.
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : 'toolu_unknown',
            content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? null),
          },
        ],
      });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') {
        blocks.push({ type: 'text', text: part.text });
        continue;
      }
      if (part.type === 'image_url' && isRecord(part.image_url)) {
        const url = typeof part.image_url.url === 'string' ? part.image_url.url : '';
        const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
        if (data !== null) {
          blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
        } else if (url !== '') {
          blocks.push({ type: 'image', source: { type: 'url', url } });
        }
        continue;
      }
      if (assistant && part.type === 'function' && typeof part.function === 'object' && part.function !== null) {
        // An OpenAI-style assistant function_call part.
        const fn = part.function as Json;
        blocks.push({ type: 'tool_use', id: typeof part.id === 'string' && part.id !== '' ? part.id : 'toolu_unknown', name: String(fn.name ?? ''), input: argumentsOf(fn.arguments) });
      }
    }
    if (blocks.length > 0) messages.push({ role: assistant ? 'assistant' : 'user', content: blocks });
  }
  // Assistant tool_calls arrive as separate tool_call entries on the message; rewrite them in.
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'assistant' || !Array.isArray(raw.tool_calls)) continue;
    const calls = raw.tool_calls.flatMap((c) => {
      if (!isRecord(c) || !isRecord(c.function)) return [];
      return [{ type: 'tool_use', id: typeof c.id === 'string' && c.id !== '' ? c.id : 'toolu_unknown', name: String(c.function.name ?? ''), input: argumentsOf(c.function.arguments) }];
    });
    for (const message of messages) {
      if (message.role === 'assistant' && Array.isArray(message.content)) {
        for (const call of calls) message.content.push(call);
        break;
      }
    }
  }
  const systemParts: string[] = [];
  for (const raw of source) {
    if (isRecord(raw) && raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
    }
  }
  const out: Json = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 4096 };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function)) return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && typeof choice.function === 'object' && choice.function !== null) {
    out.tool_choice = { type: 'tool', name: String((choice.function as Json).name ?? '') };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an internal Chat Completions answer. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const content: Json[] = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text = text === '' ? block.text : `${text}\n${block.text}`;
    if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) {
    message.tool_calls = calls;
    message.content = text === '' ? null : text;
  }
  return {
    id: typeof body.id === 'string' ? body.id : 'msg_upstream',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: finishFromStop(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0, completion_tokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0, total_tokens: 0 },
  };
}

/**
 * An Anthropic Messages SSE stream as a generator of internal chat.completion.chunk JSON objects.
 * The provider's event order: message_start, content_block_start/delta/stop..., message_delta, message_stop.
 */
export async function* openAIChunksFromAnthropicStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<Json> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const toolIds = new Map<number, string>();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      for (; at !== -1; at = buffer.indexOf('\n\n')) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          let event: unknown;
          try {
            event = JSON.parse(trimmed.slice(5).trim());
          } catch {
            continue;
          }
          if (!isRecord(event)) continue;
          if (event.type === 'message_start' && isRecord(event.message) && isRecord((event.message as Json).usage)) {
            const usage = (event.message as Json).usage as Json;
            yield { id: 'chatcmpl-up', object: 'chat.completion.chunk', created: 1767225600, model, choices: [], usage: { prompt_tokens: usage.input_tokens ?? 0, completion_tokens: 0 } };
            continue;
          }
          if (event.type === 'content_block_start' && isRecord(event.content_block) && (event.content_block as Json).type === 'tool_use') {
            const block = event.content_block as Json;
            const index = typeof event.index === 'number' ? event.index : 0;
            toolIds.set(index, typeof block.id === 'string' ? block.id : 'toolu_unknown');
            yield {
              id: 'chatcmpl-up', object: 'chat.completion.chunk', created: 1767225600, model,
              choices: [{ index: 0, delta: { tool_calls: [{ index, id: toolIds.get(index), type: 'function', function: { name: block.name ?? '', arguments: '' } }] }, finish_reason: null }],
            };
            continue;
          }
          if (event.type === 'content_block_delta' && isRecord(event.delta)) {
            const delta = event.delta as Json;
            if (delta.type === 'text_delta' && typeof delta.text === 'string') {
              yield { id: 'chatcmpl-up', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { content: delta.text }, finish_reason: null }] };
            }
            if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
              const index = typeof event.index === 'number' ? event.index : 0;
              yield {
                id: 'chatcmpl-up', object: 'chat.completion.chunk', created: 1767225600, model,
                choices: [{ index: 0, delta: { tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, finish_reason: null }],
              };
            }
            continue;
          }
          if (event.type === 'message_delta' && isRecord(event.delta)) {
            const stop = finishFromStop((event.delta as Json).stop_reason);
            const usage = isRecord(event.usage) ? event.usage : {};
            yield { id: 'chatcmpl-up', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: {}, finish_reason: stop }], usage: { completion_tokens: usage.output_tokens ?? 0 } };
            continue;
          }
          // message_stop: nothing further to emit.
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// Gemini upstream.
// ---------------------------------------------------------------------------

/** One OpenAI content part as a Gemini part; null for the rest. */
function geminiPart(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const url = typeof part.image_url.url === 'string' ? part.image_url.url : '';
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
    if (url !== '') return { fileData: { fileUri: url } };
  }
  return null;
}

/** An internal OpenAI-shaped request as a Gemini generateContent body naming `model`. */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemParts: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    if (typeof raw.content === 'string') {
      contents.push({ role, parts: [{ text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'function' && isRecord(part.function)) {
        // A model's tool call and the tool reply: emit as text so the echo's user text stays whole.
        continue;
      }
      const gem = geminiPart(part);
      if (gem !== null) parts.push(gem);
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  // Tool results (role: 'tool') appended as user text parts.
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw) || raw.role !== 'tool') continue;
    const text = typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? null);
    contents.push({ role: 'user', parts: [{ text }] });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  if (body.response_format !== undefined && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function)) return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  return out;
}

/** A Gemini generateContent answer as an internal Chat Completions answer. */
export function geminiToOpenAIAnswer(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const calls: Json[] = [];
  let callIndex = 0;
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text = text === '' ? part.text : `${text}\n${part.text}`;
    if (isRecord(part.functionCall)) {
      callIndex += 1;
      calls.push({
        id: `call_${callIndex}`,
        type: 'function',
        function: { name: String(part.functionCall.name ?? ''), arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) {
    message.tool_calls = calls;
    message.content = text === '' ? null : text;
  }
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: finishFromGemini(candidate.finishReason), logprobs: null }],
    usage: { prompt_tokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0, completion_tokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0, total_tokens: 0 },
  };
}

/** A Gemini streamGenerateContent SSE stream as a generator of internal chat.completion.chunk objects. */
export async function* openAIChunksFromGeminiStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<Json> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      for (; at !== -1; at = buffer.indexOf('\n\n')) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          let event: unknown;
          try {
            event = JSON.parse(trimmed.slice(5).trim());
          } catch {
            continue;
          }
          if (!isRecord(event)) continue;
          const candidates = Array.isArray(event.candidates) ? event.candidates : [];
          const candidate = isRecord(candidates[0]) ? candidates[0] : {};
          const content = isRecord(candidate.content) ? candidate.content : {};
          const parts = Array.isArray(content.parts) ? content.parts : [];
          for (const part of parts) {
            if (!isRecord(part) || typeof part.text !== 'string') continue;
            yield { id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { content: part.text }, finish_reason: null }] };
          }
          if (candidate.finishReason !== undefined || isRecord(event.usageMetadata)) {
            const usage = isRecord(event.usageMetadata) ? event.usageMetadata : {};
            const chunk: Json = {
              id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: 1767225600, model,
              choices: [{ index: 0, delta: {}, finish_reason: finishFromGemini(candidate.finishReason) }],
              usage: { prompt_tokens: usage.promptTokenCount ?? 0, completion_tokens: usage.candidatesTokenCount ?? 0 },
            };
            yield chunk;
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** One internal chunk as an SSE frame in the client's OpenAI wire. */
export function chunkEvent(chunk: Json): string {
  return sseEvent(null, chunk);
}
