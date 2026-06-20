// 回复 API Worker (Service Worker 格式)
// 处理用户回复的提交和读取

const ALLOWED_ORIGINS = [
  'https://doubao-whispers.pages.dev',
  'https://whisper.imagic.dpdns.org'
];

const MAX_REPLIES_PER_WINDOW = 3; // 每个 IP 每窗口最多回复数
const WINDOW_SECONDS = 300; // 时间窗口：5 分钟
const MAX_CONTENT_LENGTH = 200; // 回复内容最大长度
const MAX_NICKNAME_LENGTH = 20; // 昵称最大长度

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

// 检查频率限制
async function checkRateLimit(kv, ipHash, whisperId) {
  const key = `ratelimit:${ipHash}:${whisperId}`;
  const count = await kv.get(key, { type: 'json' }) || 0;
  
  if (count >= MAX_REPLIES_PER_WINDOW) {
    return false;
  }
  
  // 增加计数
  await kv.put(key, count + 1, { expirationTtl: WINDOW_SECONDS });
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
async function handleGet(request, kv, whisperId) {
  // 列出该 whisper 的所有回复
  const prefix = `reply:${whisperId}:`;
  const list = await kv.list({ prefix });
  
  const replies = [];
  for (const key of list.keys) {
    const reply = await kv.get(key.name, { type: 'json' });
    if (reply) {
      replies.push(reply);
    }
  }
  
  // 按时间排序
  replies.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  
  return new Response(JSON.stringify(replies), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      ...getCorsHeaders(request),
    },
  });
}

// 处理 POST 请求 - 提交回复
async function handlePost(request, kv, whisperId) {
  // 获取客户端 IP
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const ipHash = hashIp(ip);
  
  // 检查频率限制
  const allowed = await checkRateLimit(kv, ipHash, whisperId);
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
  
  // 处理回复目标
  const replyTo = (body.reply_to || '').trim();
  const replyToFloor = body.reply_to_floor ? parseInt(body.reply_to_floor) : null;
  
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
    reply_to: replyTo || '',
  };
  
  if (replyToFloor) {
    reply.reply_to_floor = replyToFloor;
  }
  
  // 存入 KV
  await kv.put(key, JSON.stringify(reply));
  
  // 返回成功响应
  const responseReply = {
    nickname,
    content,
    timestamp,
    is_doubao: false,
    reply_to: replyTo || '',
  };
  
  if (replyToFloor) {
    responseReply.reply_to_floor = replyToFloor;
  }
  
  return new Response(JSON.stringify({ 
    success: true,
    reply: responseReply,
  }), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      ...getCorsHeaders(request),
    },
  });
}

addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  const path = url.pathname;
  
  // 处理 OPTIONS 预检请求
  if (request.method === 'OPTIONS') {
    event.respondWith(handleOptions(request));
    return;
  }
  
  // 解析路径：/replies/:whisper_id
  const match = path.match(/^\/replies\/([^\/]+)\/?$/);
  if (!match) {
    event.respondWith(new Response(JSON.stringify({ error: 'Not found.' }), {
      status: 404,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    }));
    return;
  }
  
  const whisperId = decodeURIComponent(match[1]);
  const kv = REPLIES_KV;
  
  if (!kv) {
    event.respondWith(new Response(JSON.stringify({ error: 'KV namespace not configured.' }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    }));
    return;
  }
  
  if (request.method === 'GET') {
    event.respondWith(handleGet(request, kv, whisperId));
  } else if (request.method === 'POST') {
    event.respondWith(handlePost(request, kv, whisperId));
  } else {
    event.respondWith(new Response(JSON.stringify({ error: 'Method not allowed.' }), {
      status: 405,
      headers: {
        'Content-Type': 'application/json',
        'Allow': 'GET, POST, OPTIONS',
        ...getCorsHeaders(request),
      },
    }));
  }
});
