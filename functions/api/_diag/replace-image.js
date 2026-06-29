// Pages Functions: 诊断端点 —— 异步替换指定动态的配图
//
// 这是一个隐藏的诊断端点，不在前端任何页面暴露。用于一期需求：
// 替换指定 whisper 的配图。流程为异步：
//   1. 调用方 POST 本端点（带 secret 头 + whisper_id + 图片 base64）
//   2. 端点校验 secret 后，把请求写入 KV（pending_replace:{whisper_id}:{ts}）
//   3. cron runner 执行时拾取 KV 中的待处理请求，解码图片、转 webp、
//      写入 static/images/、重打包当月 tar、更新 whisper JSON，最后删除 KV key
//
// KV 绑定：IMAGE_REPLACE_KV（见 wrangler.toml）
// 认证：X-Diag-Secret 请求头需匹配环境变量 DIAG_SECRET

const ALLOWED_ORIGINS = [
  'https://doubao-whispers.pages.dev',
  'https://whisper.imagic.dpdns.org'
];

const KV_PREFIX = 'pending_replace:';
const WHISPER_ID_RE = /^\d{4}-\d{2}-\d{2}-.+$/;
const MAX_IMAGE_BYTES = 2 * 1024 * 1024; // 2MB base64 上限（解码后约 1.5MB）

function getCorsHeaders(request) {
  const origin = request.headers.get('Origin');
  const headers = {
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-Diag-Secret',
    'Access-Control-Max-Age': '86400',
  };
  if (origin && (ALLOWED_ORIGINS.includes(origin) || origin.endsWith('.pages.dev'))) {
    headers['Access-Control-Allow-Origin'] = origin;
    headers['Vary'] = 'Origin';
  } else {
    headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGINS[0];
  }
  return headers;
}

function jsonResp(status, body, corsHeaders) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders,
    },
  });
}

export async function onRequest(context) {
  const { request, env } = context;
  const corsHeaders = getCorsHeaders(request);

  // OPTIONS 预检
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (request.method !== 'POST') {
    return jsonResp(405, { error: 'Method not allowed' }, corsHeaders);
  }

  // 认证：X-Diag-Secret 必须匹配环境变量
  if (!env.DIAG_SECRET) {
    return jsonResp(500, { error: 'DIAG_SECRET not configured' }, corsHeaders);
  }
  const provided = request.headers.get('X-Diag-Secret');
  if (!provided || provided !== env.DIAG_SECRET) {
    return jsonResp(401, { error: 'Unauthorized' }, corsHeaders);
  }

  // KV 绑定检查
  if (!env.IMAGE_REPLACE_KV) {
    return jsonResp(500, { error: 'IMAGE_REPLACE_KV binding missing' }, corsHeaders);
  }

  // 解析请求体
  let body;
  try {
    body = await request.json();
  } catch (e) {
    return jsonResp(400, { error: 'Invalid JSON body' }, corsHeaders);
  }

  const whisperId = (body.whisper_id || '').trim();
  const imageBase64 = body.image_base64 || '';
  const contentType = (body.content_type || 'image/webp').trim();
  const seq = Number.isInteger(body.seq) && body.seq > 0 ? body.seq : 1;

  // 校验 whisper_id 格式 YYYY-MM-DD-slug
  if (!WHISPER_ID_RE.test(whisperId)) {
    return jsonResp(400, { error: 'Invalid whisper_id format (expected YYYY-MM-DD-slug)' }, corsHeaders);
  }

  // 校验图片数据
  if (!imageBase64) {
    return jsonResp(400, { error: 'image_base64 is required' }, corsHeaders);
  }
  if (imageBase64.length > MAX_IMAGE_BYTES) {
    return jsonResp(413, { error: 'image too large (max 2MB base64)' }, corsHeaders);
  }

  const monthStr = whisperId.slice(0, 7); // YYYY-MM
  const requestedAt = new Date().toISOString();
  // key 带时间戳，保留多次请求历史；runner 按时间升序处理，同 whisper 后到的覆盖先到的
  const key = `${KV_PREFIX}${whisperId}:${Date.now()}`;

  const value = JSON.stringify({
    whisper_id: whisperId,
    month_str: monthStr,
    image_base64: imageBase64,
    content_type: contentType,
    seq,
    requested_at: requestedAt,
  });

  try {
    await env.IMAGE_REPLACE_KV.put(key, value);
  } catch (e) {
    return jsonResp(500, { error: `KV write failed: ${e.message}` }, corsHeaders);
  }

  return jsonResp(202, {
    status: 'queued',
    key,
    whisper_id: whisperId,
    month_str: monthStr,
    seq,
    requested_at: requestedAt,
  }, corsHeaders);
}
