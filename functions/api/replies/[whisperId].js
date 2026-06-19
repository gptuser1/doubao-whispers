// Pages Functions: 回复 API
// 使用 KV REST API 访问，无需绑定

const ALLOWED_ORIGINS = [
  'https://doubao-whispers.pages.dev',
  'https://whisper.imagic.dpdns.org'
];

const MAX_REPLIES_PER_WINDOW = 3; // 每个 IP 每窗口最多回复数
const WINDOW_SECONDS = 300; // 时间窗口：5 分钟
const MAX_CONTENT_LENGTH = 200; // 回复内容最大长度
const MAX_NICKNAME_LENGTH = 20; // 昵称最大长度

// KV API 配置
const KV_API_BASE = (accountId, namespaceId) => 
  `https://api.cloudflare.com/client/v4/accounts/${accountId}/storage/kv/namespaces/${namespaceId}`;

// 简单的哈希函数，用于 IP 脱敏
function hashIp(ip) {
  let hash = 0;
  for (let i = 0; i < ip.length; i++) {
    const char = ip.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16);
}

// CORS 响应头
function getCorsHeaders(request) {
  const origin = request.headers.get('Origin');
  const headers = {
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
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

// KV API 调用封装
async function kvGet(env, key) {
  const url = `${KV_API_BASE(env.CLOUDFLARE_ACCOUNT_ID, env.KV_NAMESPACE_ID)}/values/${encodeURIComponent(key)}`;
  const response = await fetch(url, {
    headers: {
      'Authorization': `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
    },
  });
  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`KV GET failed: ${response.status}`);
  }
  return response.text();
}

async function kvPut(env, key, value, expirationTtl) {
  let url = `${KV_API_BASE(env.CLOUDFLARE_ACCOUNT_ID, env.KV_NAMESPACE_ID)}/values/${encodeURIComponent(key)}`;
  if (expirationTtl) {
    url += `?expiration_ttl=${expirationTtl}`;
  }
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
      'Content-Type': 'text/plain',
    },
    body: value,
  });
  if (!response.ok) {
    throw new Error(`KV PUT failed: ${response.status}`);
  }
  return true;
}

async function kvList(env, prefix) {
  let url = `${KV_API_BASE(env.CLOUDFLARE_ACCOUNT_ID, env.KV_NAMESPACE_ID)}/keys`;
  if (prefix) {
    url += `?prefix=${encodeURIComponent(prefix)}`;
  }
  const response = await fetch(url, {
    headers: {
      'Authorization': `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
    },
  });
  if (!response.ok) {
    throw new Error(`KV LIST failed: ${response.status}`);
  }
  const data = await response.json();
  return data.result || [];
}

// 检查频率限制
async function checkRateLimit(env, ipHash, whisperId) {
  const key = `ratelimit:${ipHash}:${whisperId}`;
  const countStr = await kvGet(env, key);
  const count = countStr ? parseInt(countStr, 10) : 0;
  
  if (count >= MAX_REPLIES_PER_WINDOW) {
    return false;
  }
  
  // 增加计数
  await kvPut(env, key, String(count + 1), WINDOW_SECONDS);
  return true;
}

// 处理 OPTIONS 预检请求
function handleOptions(request) {
  return new Response(null, {
    status: 204,
    headers: getCorsHeaders(request),
  });
}

// 处理 GET 请求 - 获取回复列表
async function handleGet(request, env, whisperId) {
  // 列出该 whisper 的所有回复
  const prefix = `reply:${whisperId}:`;
  const keys = await kvList(env, prefix);
  
  const replies = [];
  for (const keyInfo of keys) {
    const value = await kvGet(env, keyInfo.name);
    if (value) {
      try {
        const reply = JSON.parse(value);
        replies.push(reply);
      } catch (e) {
        // 忽略解析失败的
      }
    }
  }
  
  // 按时间排序
  replies.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  
  // 移除 ip_hash 等内部字段
  const publicReplies = replies.map(r => ({
    nickname: r.nickname,
    content: r.content,
    timestamp: r.timestamp,
    is_doubao: r.is_doubao || false,
  }));
  
  return new Response(JSON.stringify(publicReplies), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      ...getCorsHeaders(request),
    },
  });
}

// 处理 POST 请求 - 提交回复
async function handlePost(request, env, whisperId) {
  // 获取客户端 IP
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const ipHash = hashIp(ip);
  
  // 检查频率限制
  const allowed = await checkRateLimit(env, ipHash, whisperId);
  if (!allowed) {
    return new Response(JSON.stringify({ error: 'Too many replies, please try again later.' }), {
      status: 429,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  // 解析请求体
  let body;
  try {
    body = await request.json();
  } catch (e) {
    return new Response(JSON.stringify({ error: 'Invalid JSON body.' }), {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  // 验证内容
  const content = (body.content || '').trim();
  if (!content) {
    return new Response(JSON.stringify({ error: 'Reply content cannot be empty.' }), {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  if (content.length > MAX_CONTENT_LENGTH) {
    return new Response(JSON.stringify({ error: `Reply content too long (max ${MAX_CONTENT_LENGTH} chars).` }), {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  // 处理昵称
  const nickname = (body.nickname || '').trim() || '匿名网友';
  if (nickname.length > MAX_NICKNAME_LENGTH) {
    return new Response(JSON.stringify({ error: `Nickname too long (max ${MAX_NICKNAME_LENGTH} chars).` }), {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  // 生成唯一 ID
  const uuid = crypto.randomUUID();
  const key = `reply:${whisperId}:${uuid}`;
  
  // 生成时间戳（北京时间）
  const now = new Date();
  const timestamp = new Date(now.getTime() + 8 * 60 * 60 * 1000).toISOString().replace('Z', '+08:00');
  
  // 构建回复对象
  const reply = {
    nickname,
    content,
    timestamp,
    ip_hash: ipHash,
    is_doubao: false,
  };
  
  // 存入 KV
  await kvPut(env, key, JSON.stringify(reply));
  
  // 返回成功响应
  return new Response(JSON.stringify({ 
    success: true,
    reply: {
      nickname,
      content,
      timestamp,
      is_doubao: false,
    }
  }), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      ...getCorsHeaders(request),
    },
  });
}

// 主处理函数
export async function onRequest(context) {
  const { request, env, params } = context;
  
  // 处理 OPTIONS 预检请求
  if (request.method === 'OPTIONS') {
    return handleOptions(request);
  }
  
  const whisperId = params.whisperId;
  if (!whisperId) {
    return new Response(JSON.stringify({ error: 'Not found.' }), {
      status: 404,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  // 检查必要的环境变量
  if (!env.CLOUDFLARE_API_TOKEN || !env.CLOUDFLARE_ACCOUNT_ID || !env.KV_NAMESPACE_ID) {
    return new Response(JSON.stringify({ error: 'Server configuration error.' }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  if (request.method === 'GET') {
    return handleGet(request, env, whisperId);
  } else if (request.method === 'POST') {
    return handlePost(request, env, whisperId);
  } else {
    return new Response(JSON.stringify({ error: 'Method not allowed.' }), {
      status: 405,
      headers: {
        'Content-Type': 'application/json',
        'Allow': 'GET, POST, OPTIONS',
        ...getCorsHeaders(request),
      },
    });
  }
}
