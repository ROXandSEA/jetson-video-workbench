'use strict';
const $ = id => document.getElementById(id);
const activeStatuses = new Set(['loading', 'running', 'encoding', 'evaluating', 'stopping']);
let token = '', selectedMode = 'uav', state = { status: 'idle' }, config = null;
let source = null, lastSeq = -1, displayedJob = null, resultShown = null, starting = false;
let connected = false, picking = false, frameBusy = false, displayedFrameURL = null;
const video = $('video');

async function api(path, data) {
  const options = data === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Studio-Token': token }, body: JSON.stringify(data)
  };
  const response = await fetch(path, { ...options, cache: 'no-store' });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '请求未完成。');
  return result;
}
function error(message) { $('error-text').textContent = message; $('error-message').hidden = false; }
function clearError() { $('error-message').hidden = true; }
function timecode(seconds) {
  if (!Number.isFinite(seconds)) return '00:00';
  seconds = Math.max(0, Math.floor(seconds));
  const h = Math.floor(seconds / 3600), m = Math.floor(seconds / 60) % 60, s = seconds % 60;
  return (h ? String(h).padStart(2, '0') + ':' : '') + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}
function persist(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }
function saved(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
function engineReady() { return config && config.engines.some(e => e.mode === selectedMode && e.exists); }
function canRun() { return connected && token && !starting && !picking && !activeStatuses.has(state.status) && $('source-path').value.trim() && $('output-path').value.trim(); }
function updateButtons() {
  const active = activeStatuses.has(state.status);
  $('start-button').disabled = !canRun();
  for (const id of ['source-path', 'output-path', 'pick-source', 'pick-output', 'empty-pick', 'annotation-path', 'profile']) $(id).disabled = active || starting || picking;
  document.querySelectorAll('.mode').forEach(button => { button.disabled = active || starting; });
  $('stop-button').hidden = !active;
  $('stop-button').disabled = state.status === 'stopping';
  $('open-output').hidden = !state.output_dir || active;
}
function setMode(mode) {
  selectedMode = mode;
  document.querySelectorAll('.mode').forEach(button => {
    const chosen = button.dataset.mode === mode;
    button.classList.toggle('active', chosen); button.setAttribute('aria-pressed', String(chosen));
  });
  renderProfiles(mode);
  persist('trackform-mode', mode);
  renderEvaluation(null, 'idle');
  updateConnection(); updateButtons();
}
function renderProfiles(mode) {
  if (!config) return;
  const select = $('profile'); select.replaceChildren();
  for (const [id, item] of Object.entries(config.profiles[mode] || {})) {
    const option = document.createElement('option'); option.value = id; option.textContent = item.label; select.append(option);
  }
  select.value = config.default_profiles[mode] || 'balanced';
  renderProfileNote();
}
function renderProfileNote() {
  const item=config?.profiles?.[selectedMode]?.[$('profile').value];
  $('profile-note').textContent=item?.description || '每帧检测并保存。小目标优先选 1280；640 速度更快。';
}
function updateConnection() {
  $('connection-dot').classList.toggle('connected', connected);
  $('connection-label').textContent = !connected ? '本地服务未连接' : engineReady() ? '本地已连接 · 引擎已配置' : '本地已连接 · 待配置引擎';
}
function showVideo(url, label, autoplay = false) {
  $('live-frame').hidden = true;
  $('empty-state').hidden = true;
  video.hidden = false;
  video.src = url;
  video.load();
  $('view-title').textContent = label;
  $('view-caption').textContent = label === '追踪结果' ? 'TRACKED. SAVED. READY TO REPLAY.' : 'SOURCE / ORIGINAL FOOTAGE';
  $('viewer-badge').hidden = true;
  $('frame-counter').hidden = true;
  if (autoplay) video.play().catch(() => {});
}
async function setSource(value) {
  source = value;
  renderEvaluation(null, 'idle');
  $('source-path').value = value.path;
  $('file-note').textContent = value.name + ' · ' + (value.bytes / 1048576).toFixed(1) + ' MB';
  persist('trackform-source', value.path);
  lastSeq = -1; displayedJob = null;
  // Keep polling the previous report without replacing a newly selected source.
  resultShown = activeStatuses.has(state.status) ? null : state.id;
  if (!activeStatuses.has(state.status)) showVideo(value.url, '原始视频');
  updateButtons();
}
async function registerSource() {
  const path = $('source-path').value.trim();
  if (!path) return;
  if (source && source.path === path) return source;
  const result = await api('/api/source', { path });
  await setSource(result);
  return result;
}
async function pick(kind) {
  if (picking || activeStatuses.has(state.status)) return;
  picking = true; updateButtons(); clearError();
  try {
    const result = await api('/api/pick/' + kind, {});
    if (result.cancelled) return;
    if (kind === 'video') await setSource(result);
    else { $('output-path').value = result.path; persist('trackform-output', result.path); }
  } catch (exc) { error(exc.message); }
  finally { picking = false; updateButtons(); }
}
function renderSettings() {
  if (!config) return;
  $('engine-list').replaceChildren();
  for (const engine of config.engines) {
    const card = document.createElement('div'); card.className = 'engine-card';
    const top = document.createElement('div'), title = document.createElement('span'), status = document.createElement('small'), path = document.createElement('code');
    title.textContent = engine.mode === 'uav' ? '无人机 / UAV' : '多类别 / COCO 80';
    status.textContent = engine.exists ? '文件已配置 · 运行待验证' : '待构建 / 配置';
    if (engine.exists) status.className = 'ready';
    path.textContent = engine.path;
    top.append(title, status); card.append(top, path); $('engine-list').append(card);
  }
  const missing = Object.entries(config.dependencies).filter(([, exists]) => !exists).map(([name]) => name);
  $('dependency-label').textContent = missing.length ? '待安装：' + missing.join(' / ') : '基础模块已安装 · CUDA 运行待验证';
  $('ffmpeg-label').textContent = config.ffmpeg ? 'FFmpeg 已安装 · 编码能力待运行验证' : '待安装 FFmpeg';
}
function updateTransport() {
  const enabled = !video.hidden && Number.isFinite(video.duration) && video.duration > 0;
  $('play-button').disabled = !enabled;
  $('seek').disabled = !enabled;
  $('replay-button').disabled = !enabled;
  $('current-time').textContent = timecode(video.currentTime);
  $('duration').textContent = timecode(video.duration);
  $('seek').value = enabled ? Math.round(video.currentTime / video.duration * 1000) : 0;
  const ns = 'http://www.w3.org/2000/svg';
  const icon = $('play-icon'); icon.replaceChildren();
  const path = document.createElementNS(ns, 'path');
  path.setAttribute('d', video.paused || video.hidden ? 'm8 5 11 7-11 7Z' : 'M6 5h4v14H6Zm8 0h4v14h-4Z'); icon.append(path);
  $('play-button').setAttribute('aria-label', video.paused ? '播放视频' : '暂停视频');
}
async function refreshFrame(current) {
  if (!current.id || current.preview_seq <= 0 || frameBusy) return;
  if (displayedJob === current.id && lastSeq === current.preview_seq) return;
  frameBusy = true;
  const id = current.id, seq = current.preview_seq;
  try {
    const response = await fetch(`/api/frame/${id}?v=${seq}`, { cache: 'no-store' });
    if (!response.ok) return;
    const blob = await response.blob();
    // Discard a frame if a newer task or result was displayed while fetching.
    if (state.id !== id || resultShown === id || source && !activeStatuses.has(state.status) && displayedJob === null) return;
    const url = URL.createObjectURL(blob);
    const previous = displayedFrameURL;
    displayedFrameURL = url;
    $('live-frame').src = url;
    $('live-frame').hidden = false;
    video.pause(); video.hidden = true; $('empty-state').hidden = true;
    if (previous) URL.revokeObjectURL(previous);
    displayedJob = id; lastSeq = seq;
    $('view-title').textContent = '追踪画面';
    $('view-caption').textContent = current.mode === 'coco' ? 'COCO 80 / DETECTION + ID TRACKING' : 'UAV / DETECTION + ID TRACKING';
    $('frame-counter').hidden = false;
    updateTransport();
  } catch (_) { /* a transient preview failure must not interrupt inference */ }
  finally { frameBusy = false; }
}
function renderEvaluation(evaluation, status) {
  const complete = evaluation && evaluation.status === 'completed';
  const score = value => value === null || value === undefined || value < 0 ? '—' : (value * 100).toFixed(2) + '%';
  $('map50').textContent = complete ? score(evaluation.map50) : '—';
  $('map50-95').textContent = complete ? score(evaluation.map50_95) : '—';
  $('evaluation-frames').textContent = complete ? String(evaluation.frames) : '—';
  $('evaluation-objects').textContent = complete ? String(evaluation.valid_objects) : '—';
  $('evaluation-status').textContent = status === 'evaluating' ? '正在评估' : complete ? '已评估' : evaluation ? evaluation.status === 'invalid' ? '标注不适用' : '无法评估' : '等待评估';
  $('evaluation-message').textContent = evaluation ? evaluation.message : '完成检测后，用配套人工标注计算。没有标注时不显示分数。';
  const scoredClasses = complete ? (evaluation.per_class || []).filter(c => c.map50_95 !== null).map(c => c.name) : [];
  const absentClasses = complete ? (evaluation.per_class || []).filter(c => c.map50_95 === null).map(c => c.name) : [];
  $('evaluation-scope').textContent = complete
    ? `范围：${evaluation.scope} · 计分类别：${scoredClasses.join(' / ')}${absentClasses.length ? ' · 无有效标注：' + absentClasses.join(' / ') + '（不参与平均）' : ''} · 置信度 ≥ ${evaluation.confidence_min} · 评估每帧每类最多 100 个框。仅代表当前视频的检测 AP。`
    : '评估原始检测框。mAP 不代表跟踪 ID 准确率，置信度也不是准确率。';
  const classes = $('evaluation-classes');
  classes.replaceChildren(); classes.hidden = !complete;
  if (complete) for (const category of evaluation.per_class || []) {
    const item = document.createElement('span');
    item.textContent = `${category.name} AP@.50:.95 ${score(category.map50_95)}`;
    classes.append(item);
  }
}
function render(current) {
  const prior = state; state = current;
  const active = activeStatuses.has(current.status);
  if (active && current.mode && current.mode !== selectedMode) setMode(current.mode);
  if (active && current.profile) $('profile').value = current.profile;
  renderProfileNote();
  if (current.id && current.id !== prior.id) { lastSeq = -1; displayedJob = current.id; resultShown = null; }
  $('task-message').textContent = current.message;
  $('percent').textContent = current.progress !== null ? `${Math.round(current.progress)}%` : active ? '处理中' : current.status === 'idle' ? '待开始' : current.status === 'failed' ? '未完成' : '已停止';
  $('progress-fill').style.width = Math.max(0, Math.min(100, current.progress || 0)) + '%';
  document.querySelector('.progress-track').classList.toggle('indeterminate', active && current.progress === null);
  $('fps').textContent = current.fps === null ? '—' : Number(current.fps).toFixed(1);
  $('targets').textContent = current.status === 'idle' ? '—' : String(current.targets);
  $('elapsed-label').textContent = current.elapsed ? timecode(current.elapsed) + ' 已用时' : '—';
  $('live-dot').classList.toggle('on', active);
  $('viewer-badge').hidden = !(active && current.preview_seq > 0);
  $('viewer-badge-label').textContent = current.status === 'encoding' ? '正在保存' : current.status === 'stopping' ? '正在停止' : '处理中';
  $('frame-number').textContent = String(current.frame).padStart(6, '0');
  $('loading-overlay').hidden = current.status !== 'loading';
  $('loading-text').textContent = current.message;
  if (current.result_url && resultShown !== current.id) {
    resultShown = current.id;
    showVideo(current.result_url, '追踪结果', true);
  } else if (!current.result_url && current.preview_seq > 0) {
    refreshFrame(current);
  }
  if (current.status === 'failed' && (prior.status !== 'failed' || prior.id !== current.id)) error(current.error || current.message);
  const matchingSource = current.source_path === $('source-path').value.trim() && current.mode === selectedMode && current.profile === $('profile').value;
  renderEvaluation(matchingSource ? current.evaluation : null, current.status);
  updateButtons();
}

$('pick-source').addEventListener('click', () => pick('video'));
$('empty-pick').addEventListener('click', () => pick('video'));
$('pick-output').addEventListener('click', () => pick('output'));
$('profile').addEventListener('change', () => {renderProfileNote();renderEvaluation(null, 'idle');});
$('source-path').addEventListener('input', () => { source = null; renderEvaluation(null, 'idle'); updateButtons(); });
$('source-path').addEventListener('change', () => { if ($('source-path').value.trim()) registerSource().catch(exc => error(exc.message)); });
$('output-path').addEventListener('input', () => { persist('trackform-output', $('output-path').value.trim()); updateButtons(); });
document.querySelectorAll('.mode').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
$('start-button').addEventListener('click', async () => {
  if (!canRun()) return;
  starting = true; updateButtons(); clearError();
  try {
    const input = await registerSource();
    if (!input) throw new Error('请先选择输入视频。');
    const current = await api('/api/start', { source: input.path, output: $('output-path').value.trim(), mode: selectedMode, profile: $('profile').value, annotations: $('annotation-path').value.trim() });
    video.pause(); resultShown = null; displayedJob = current.id;
    render(current);
  } catch (exc) { error(exc.message); }
  finally { starting = false; updateButtons(); }
});
$('stop-button').addEventListener('click', async () => { try { render(await api('/api/stop', {})); } catch (exc) { error(exc.message); } });
$('open-output').addEventListener('click', async () => { try { await api('/api/open-output', {}); } catch (exc) { error(exc.message); } });
$('dismiss-error').addEventListener('click', clearError);
$('play-button').addEventListener('click', () => { if (video.paused) video.play().catch(() => error('视频无法播放，请检查输入编码。')); else video.pause(); });
$('seek').addEventListener('input', () => { if (Number.isFinite(video.duration)) video.currentTime = Number($('seek').value) / 1000 * video.duration; });
$('replay-button').addEventListener('click', () => { video.currentTime = 0; video.play().catch(() => {}); });
$('fullscreen-button').addEventListener('click', () => {
  const operation = document.fullscreenElement ? document.exitFullscreen() : $('viewer').requestFullscreen();
  operation.catch(() => error('当前浏览器无法进入全屏。'));
});
for (const event of ['loadedmetadata', 'timeupdate', 'play', 'pause', 'ended', 'emptied']) video.addEventListener(event, updateTransport);
video.addEventListener('loadedmetadata', () => { $('resolution-label').textContent = `${video.videoWidth} × ${video.videoHeight}`; });
$('live-frame').addEventListener('load', () => { const img = $('live-frame'); $('resolution-label').textContent = `${img.naturalWidth} × ${img.naturalHeight}`; });
video.addEventListener('error', () => { if (video.getAttribute('src')) error('浏览器无法回放这个视频编码。原视频仍可用于处理；结果 MP4 使用 H.264 编码。'); });
$('settings-button').addEventListener('click', () => { renderSettings(); $('settings-dialog').showModal(); });
$('close-settings').addEventListener('click', () => $('settings-dialog').close());
$('settings-dialog').addEventListener('click', event => { if (event.target === $('settings-dialog')) { const r = event.target.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close(); } });
$('refresh-config').addEventListener('click', async () => {
  try { config = await api('/api/bootstrap'); token = config.token; connected = true; renderSettings(); updateConnection(); }
  catch (exc) { error(exc.message); }
});
async function poll() {
  try { const current = await api('/api/status'); connected = true; render(current); }
  catch (_) { connected = false; updateButtons(); }
  updateConnection();
  setTimeout(poll, activeStatuses.has(state.status) ? 125 : 800);
}
async function init() {
  try {
    config = await api('/api/bootstrap'); token = config.token; connected = true;
    renderProfiles(selectedMode);
    const mode = saved('trackform-mode') || config.state.mode; if (mode === 'coco' || mode === 'uav') setMode(mode);
    $('output-path').value = saved('trackform-output') || config.default_output;
    render(config.state); updateConnection(); renderSettings();
    const oldSource = saved('trackform-source') || config.state.source_path;
    if (oldSource === config.state.source_path && selectedMode === config.state.mode && config.state.profile) $('profile').value = config.state.profile;
    if (oldSource) { $('source-path').value = oldSource; if (state.status === 'idle') { try { await registerSource(); } catch (_) { source = null; } } }
    updateButtons();
  } catch (exc) { error('本地服务未连接，请用 python server.py 启动后打开本机地址。'); updateConnection(); }
  poll();
}
init();
