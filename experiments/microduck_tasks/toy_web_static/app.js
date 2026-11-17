"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="microduck-session"]').content;
const input = $("chat-input");
let composing = false, compositionEnded = 0, conversationKey = "", lastFrame = -1, lastEpisode = -1;
const stateNames = {idle:"空闲",running:"执行中",paused:"已安全暂停",failed:"任务未完成"};
const actionNames = {stand:"站立",nod:"点头",look_around:"左右看",sit_stand:"坐下再站起",ground_pick:"俯身再起",roll:"前滚翻",flamingo_left:"抬左脚",flamingo_right:"抬右脚",moonwalk:"后退滑步",body_sway:"轻晃身体",open_beak:"张嘴",close_beak:"闭嘴",groove:"节奏舞步",caught_dancing:"害羞舞步",turn_left:"左转",turn_right:"右转",move_forward:"前进",move_backward:"后退",move_left:"左移",move_right:"右移",kick_left:"左脚踢球",kick_right:"右脚踢球",sit_hold:"保持坐姿",sit_down:"坐下",stand_up:"站起"};
const directions = {left:"左",right:"右",forward:"前",backward:"后"};
function goalLabel(goal) {
  if (!goal) return "暂无执行中的任务";
  if (goal.type === "perform") return `${actionNames[goal.action] || goal.action} × ${goal.repeat}`;
  if (goal.type === "posture") return goal.pose === "sitting" ? "坐下并保持" : "站起来";
  if (goal.type === "turn") return `向${directions[goal.direction]}转 ${goal.angle_deg}°`;
  if (goal.type === "move") return `向${directions[goal.direction]}走 · ${goal.limit.kind === "until_cancelled" ? "持续运动" : goal.limit.seconds + " 秒"}`;
  const color = (goal.target?.attributes || []).find((a) => a.name === "color")?.value;
  const target = {red:"红球",blue:"蓝球",yellow:"黄球"}[color] || "目标";
  return goal.type === "kick" ? `踢${target}` : `走到${target}旁边`;
}
function notice(message) { $("notice").hidden = !message; $("notice").textContent = message || ""; }
async function post(path, body) {
  const response = await fetch(path, {method:"POST",headers:{"Content-Type":"application/json","X-MicroDuck-Session":token},body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败，请重试。");
  return result;
}
async function send(text) {
  if (!text.trim()) return false;
  try { await post("/api/message",{text}); notice(""); await refresh(); return true; }
  catch (error) { notice(error.message); return false; }
}
input.addEventListener("compositionstart", () => { composing = true; });
input.addEventListener("compositionend", () => { composing = false; compositionEnded = performance.now(); });
input.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  if (composing || event.isComposing || event.keyCode === 229 || performance.now() - compositionEnded < 80) return;
  event.preventDefault(); $("chat-form").requestSubmit();
});
$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (composing || $("send").disabled) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  if (!await send(text) && !input.value) input.value = text;
  input.focus();
});
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { input.value = button.dataset.prompt; input.focus(); }));
$("pause").addEventListener("click", () => send("暂停"));
$("resume").addEventListener("click", () => send("继续"));
$("stop").addEventListener("click", () => send("停下"));
$("reset").addEventListener("click", async () => { try { await post("/api/reset",{}); notice(""); } catch (error) { notice(error.message); } });
function message(role, text, decision) {
  const row = document.createElement("div"); row.className = `message ${role}${text == null ? " pending" : ""}`;
  const label = document.createElement("span"); label.className = "message-label"; label.textContent = role === "user" ? "你" : "小鸭 · 程序回复";
  const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = text ?? "正在处理…";
  row.append(label, bubble);
  if (decision) { const badge = document.createElement("span"); badge.className = "decision"; badge.textContent = decision; row.append(badge); }
  return row;
}
const welcome = $("messages").innerHTML;
function renderMessages(messages, episode) {
  const key = JSON.stringify([episode,messages]); if (key === conversationKey) return;
  conversationKey = key;
  const panel = $("messages"); const nearBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 80;
  panel.replaceChildren();
  if (!messages.length) {
    panel.innerHTML = welcome;
    panel.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { input.value = button.dataset.prompt; input.focus(); }));
  } else {
    for (const item of messages) { panel.append(message("user",item.user),message("assistant",item.reply,item.decision)); }
  }
  if (nearBottom || messages.at(-1)?.reply == null) panel.scrollTop = panel.scrollHeight;
}
function update(state) {
  const ready = state.phase === "ready";
  $("connection-dot").classList.toggle("connected",ready);
  $("connection").textContent = ready ? "已连接" : state.phase === "error" ? "需要处理" : "准备中";
  ["chat-input","send","pause","stop","reset"].forEach((id) => { $(id).disabled = !ready; });
  const execution = state.execution, pending = execution.pending || [];
  $("resume").disabled = !ready || !(execution.state === "paused" || state.pause_requested);
  $("execution-state").textContent = state.pause_requested && execution.state === "running" ? "等待安全暂停" : stateNames[execution.state] || execution.state;
  $("queue-count").textContent = `${pending.length + (execution.active ? 1 : 0)} / ${state.max_goals}`;
  $("active-goal").textContent = goalLabel(execution.active);
  $("last-result").hidden = !execution.last_result;
  if (execution.last_result) {
    const result = execution.last_result;
    $("last-result").textContent = `上一步：${goalLabel(result.goal)} · ${{succeeded:"已完成",failed:"未通过验收",cancelled:"已取消",blocked:"已暂停",paused:"已安全暂停"}[result.status] || result.status}`;
  }
  $("pending-count").textContent = pending.length;
  $("queue-details").hidden = pending.length === 0;
  $("pending-goals").replaceChildren(...pending.map((goal) => { const li = document.createElement("li"); li.textContent = goalLabel(goal); return li; }));
  $("pause-hint").textContent = state.pause_requested ? "已请求安全暂停：保留推理，等待动作收尾；持续行走会先收稳。" : "暂停保留推理和任务，等待当前完整动作结束后挂起。";
  const telemetry = state.telemetry;
  $("clock").textContent = `${(telemetry.time_s || 0).toFixed(2)} s`;
  $("tilt").textContent = telemetry.tilt_deg == null ? "—" : `${telemetry.tilt_deg.toFixed(1)}°`;
  $("stationary").textContent = telemetry.stationary == null ? "—" : telemetry.stationary ? "静止" : "运动中";
  $("skill").textContent = actionNames[telemetry.skill] || telemetry.skill || "—";
  $("scene-state").textContent = telemetry.failure ? `执行异常：${telemetry.failure}` : !ready ? state.detail : state.busy ? "正在处理指令 · 推理时画面暂时保持" : stateNames[execution.state];
  const teacher = state.backend === "toy_rule_teacher";
  $("backend-label").textContent = teacher ? "规则教师 · 执行器测试" : "Q1 · checkpoint 100 · 旧权重，新上限 15";
  $("model-chip").textContent = teacher ? "规则" : "Q1";
  $("model-note").textContent = teacher ? "当前为规则教师，只测试执行器与页面交互，不代表 Q1 模型效果。" : "Q1 生成任务计划；下方自然语言回复由程序模板提供，尚未训练自由聊天。";
  $("loading").hidden = ready && state.frame_id > 0;
  $("load-title").textContent = state.phase === "error" ? "初始化未完成" : state.phase === "resetting" ? "正在重置场景" : "正在唤醒小鸭";
  $("load-detail").textContent = state.detail;
  if (state.frame_id && (lastFrame !== state.frame_id || lastEpisode !== state.episode)) {
    for (const name of ["world","front","near"]) { $(name).src = `/frame/${name}.jpg?v=${state.episode}-${state.frame_id}`; $(name).hidden = false; }
    lastFrame = state.frame_id; lastEpisode = state.episode;
  }
  renderMessages(state.messages,state.episode);
  if (state.errors.length) notice(state.errors.at(-1));
}
let fetching = false;
async function refresh() {
  if (fetching) return;
  fetching = true;
  try { const response = await fetch("/api/state"); if (!response.ok) throw new Error("本地服务连接失败"); update(await response.json()); }
  catch (error) { $("connection").textContent = "连接中断"; $("connection-dot").classList.remove("connected"); notice(error.message); }
  finally { fetching = false; }
}
refresh(); setInterval(refresh,180);
