// Pages Functions: 回复 API (D1 版本)
const ALLOWED_ORIGINS = [
  'https://doubao-whispers.pages.dev',
  'https://whisper.imagic.dpdns.org'
];
const MAX_REPLIES_PER_WINDOW = 3; // 每个 IP 每窗口最多回复数
const WINDOW_MINUTES = 5; // 时间窗口：5 分钟
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
async function checkRateLimit(db, ipHash, whisperId) {
  const result = await db.prepare(
    `SELECT COUNT(*) as count FROM replies 
     WHERE ip_hash = ? AND whisper_id = ? 
     AND created_at > datetime('now', ?)`
  ).bind(ipHash, whisperId, `-${WINDOW_MINUTES} minutes`).first();
  
  return result.count < MAX_REPLIES_PER_WINDOW;
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
  const db = env.DB;

  const result = await db.prepare(
    `SELECT * FROM replies WHERE whisper_id = ? ORDER BY timestamp ASC`
  ).bind(whisperId).all();

  const replies = result.results || [];

  // 角色昵称 -> author ID 反查表。is_doubao=1 的行一定是系统写入的角色回复，
  // 6 个角色昵称唯一且受控，借此补出 author 字段供前端区分样式/头像。
  // （replies 表未存 author_id 列，故用昵称反查，避免 DB 迁移。）
  const NICKNAME_TO_AUTHOR = {
    '豆包': 'doubao',
    '咕嘎': 'guga',
    'Doro': 'doro',
    '菲比': 'feibi',
    '白子': 'baizi',
    '糯糯': 'nuonuo',
  };

  // 构建公开的回复对象（移除内部字段）
  const publicReplies = replies.map(r => {
    const isChar = r.is_doubao === 1;
    const reply = {
      nickname: r.nickname,
      content: r.content,
      timestamp: r.timestamp,
      is_doubao: isChar,
      // 角色回复补出 author（用户回复为空字符串，与仓库 data/replies 一致）
      author: isChar ? (NICKNAME_TO_AUTHOR[r.nickname] || '') : '',
      floor: r.floor,
    };
    if (r.reply_to) {
      reply.reply_to = r.reply_to;
    }
    if (r.reply_to_floor) {
      reply.reply_to_floor = r.reply_to_floor;
    }
    return reply;
  });
  
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
  const db = env.DB;
  
  // 获取客户端 IP
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const ipHash = hashIp(ip);
  
  // 检查频率限制
  const allowed = await checkRateLimit(db, ipHash, whisperId);
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
  
  // 生成时间戳（北京时间）
  const now = new Date();
  const timestamp = new Date(now.getTime() + 8 * 60 * 60 * 1000).toISOString().replace('Z', '+08:00');
  
  // 处理 reply_to 和 reply_to_floor
  const reply_to = body.reply_to ? String(body.reply_to).trim() : '';
  const reply_to_floor = body.reply_to_floor ? parseInt(body.reply_to_floor, 10) : null;
  
  // 插入回复（楼层号用子查询自动计算）
  const insertResult = await db.prepare(
    `INSERT INTO replies (whisper_id, nickname, content, timestamp, reply_to, reply_to_floor, floor, ip_hash, is_doubao)
     VALUES (?, ?, ?, ?, ?, ?, (SELECT COALESCE(MAX(floor), 0) + 1 FROM replies WHERE whisper_id = ?), ?, 0)`
  ).bind(
    whisperId,
    nickname,
    content,
    timestamp,
    reply_to || null,
    reply_to_floor || null,
    whisperId,
    ipHash
  ).run();
  
  if (!insertResult.success) {
    return new Response(JSON.stringify({ error: 'Failed to post reply.' }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        ...getCorsHeaders(request),
      },
    });
  }
  
  // 获取刚插入的回复的楼层号
  const newReply = await db.prepare(
    `SELECT floor FROM replies WHERE rowid = ?`
  ).bind(insertResult.meta.last_row_id).first();
  
  // 构建返回的回复对象
  const publicReply = {
    nickname,
    content,
    timestamp,
    is_doubao: false,
    floor: newReply ? newReply.floor : 0,
  };
  if (reply_to) {
    publicReply.reply_to = reply_to;
  }
  if (reply_to_floor) {
    publicReply.reply_to_floor = reply_to_floor;
  }
  
  // 返回成功响应
  return new Response(JSON.stringify({ 
    success: true,
    reply: publicReply
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
  
  // 检查 D1 绑定
  if (!env.DB) {
    return new Response(JSON.stringify({ error: 'Server configuration error: DB binding not found.' }), {
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
