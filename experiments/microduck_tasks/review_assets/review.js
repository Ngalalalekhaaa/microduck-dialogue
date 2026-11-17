"use strict";
(() => {
 const $ = id => document.getElementById(id), data = window.REVIEW_DATA;
 if (!data) { $('loading').textContent='数据加载失败：请保留同目录 review_data.js 后重新打开。'; return; }
 const records=data.records, byId=new Map(records.map(r=>[r.sample_id,r]));
 const taskKey=r=>r.episode_id+'/'+r.task_id, tasks=new Map();
 for (const r of records) { const key=taskKey(r); if(!tasks.has(key))tasks.set(key,[]);tasks.get(key).push(r); }
 for (const a of tasks.values())a.sort((x,y)=>x.source.event_id.localeCompare(y.source.event_id));
 let selected=[], index=0, activeTask=null, timer=null, marks={};
 try {marks=JSON.parse(localStorage.getItem('microduck-review-20261115')||'{}');}catch(_e){}
 const text=(tag,value,cls)=>{const el=document.createElement(tag);el.textContent=value;if(cls)el.className=cls;return el;};
 const pretty=x=>JSON.stringify(x,null,2), current=()=>selected[index];
 const skill=r=>r.target.skill||r.target.type;
 const phases={search:'搜索',face:'朝向调整',approach:'接近',align:'对齐',settle:'站稳',kick:'踢球',done:'结束 / 回复',direct:'直接动作',avoid:'避让',recover:'固定恢复',interpret:'开始'};
 const fill=(id,items,label=x=>x)=>{for(const value of [...new Set(items)].sort()){const o=text('option',label(value));o.value=value;$(id).append(o);}};
 fill('phase',records.map(r=>r.review.phase),x=>(phases[x]||x)+' / '+x); fill('skill',records.map(skill));
 function chart(id,counts){const max=Math.max(...Object.values(counts));for(const [name,n]of Object.entries(counts).sort((a,b)=>b[1]-a[1])){const row=text('div','','chart');row.append(text('span',name));const b=text('div','','bar'),i=document.createElement('i');i.style.width=(100*n/max)+'%';b.append(i);row.append(b,text('span',n));$(id).append(row);}}
 chart('skillChart',data.counts.train.skills);chart('phaseChart',data.counts.train.rule_phases_audit_only);
 function resetFilters(){for(const id of ['split','phase','skill','color','search'])$(id).value='';}
 function jump(r){if(!r)return;stop();activeTask=taskKey(r);$('mode').value='task';resetFilters();filter(r.sample_id);}
 function highlightButton(label,r){if(!r)return;const b=text('button',label);b.onclick=()=>jump(r);$('highlights').append(b);}
 for(const h of (data.highlights||[])){highlightButton(h.title||h.name||h.sample_id,byId.get(h.sample_id));}
 for(const idx of [161,1019,991,368])highlightButton('模型错误 · val['+idx+']',records.find(r=>r.review.split==='val'&&r.review.index===idx));
 for(const reason of ['verification_failed','not_found','timeout'])highlightButton('失败轨迹 · '+reason,records.find(r=>r.review.exclusion==='task_not_successful:'+reason));
 function stop(){clearInterval(timer);timer=null;$('play').textContent='每秒一帧';}
 function filter(id){
  stop();const mode=$('mode').value,q=$('search').value.trim().toLowerCase();
  selected=records.filter(r=>{
   const rv=r.review;
   if(mode==='errors'&&(!rv.included||!Object.values(rv.models).some(x=>!x.matched)))return false;
   if(mode==='all'&&!rv.included)return false;if(mode==='excluded'&&rv.included)return false;
   if(mode==='task'&&taskKey(r)!==activeTask)return false;if(mode==='marked'&&!marks[r.sample_id])return false;
   if($('split').value&&rv.split!==$('split').value)return false;
   if($('phase').value&&rv.phase!==$('phase').value)return false;
   if($('skill').value&&skill(r)!==$('skill').value)return false;
   if($('color').value&&(r.target.target_color||'none')!==$('color').value)return false;
   if(q&&!(r.sample_id+' '+r.instruction+' '+r.scene_seed).toLowerCase().includes(q))return false;
   return true;
  });index=Math.max(0,id?selected.findIndex(r=>r.sample_id===id):0);render();
 }
 function render(){
  $('loading').textContent=selected.length?'当前筛选 '+selected.length+' 条；浏览器包含正式数据 '+(data.split_rows.train+data.split_rows.val+data.split_rows.test)+' 条，以及 '+data.split_rows.excluded+' 条失败原始事件。':'没有匹配记录，请清除筛选。';
  $('content').hidden=!selected.length;$('position').textContent=selected.length?(index+1)+' / '+selected.length:'0 / 0';
  $('slider').max=Math.max(0,selected.length-1);$('slider').value=index;$('prev').disabled=index===0;$('next').disabled=index>=selected.length-1;
  if(!selected.length)return;const r=current(),rv=r.review;
  $('instruction').textContent=r.instruction;$('sourceid').textContent=r.sample_id+' · seed '+r.scene_seed+' · '+r.task_id+' · '+(rv.included?rv.split+'.jsonl 第 '+rv.line+' 行':'原始事件第 '+rv.line+' 行');
  $('badges').replaceChildren(...[rv.split,rv.included?'正式数据':'未参与训练',phases[rv.phase]||rv.phase,'step '+(r.context.step??'—')].map(s=>text('span',s,'pill')));
  let risk=rv.included?'':'这是失败轨迹的实际动作，未进入训练集，也不是纠正后的正确标签。排除原因：'+rv.exclusion;
  if(rv.highlight)risk+=(risk?'\n':'')+(rv.highlight.risk_explanation||rv.highlight.risk||rv.highlight.note||rv.highlight.reason||JSON.stringify(rv.highlight));
  $('risk').hidden=!risk;$('risk').textContent=risk;
  for(const camera of ['front','near']){$(camera).src=rv.images[camera];$(camera+'Link').href=rv.images[camera];}
  $('targetTitle').textContent=rv.included?'规则示范标签（训练目标）':'失败轨迹的原始动作（未训练）';$('target').textContent=pretty(r.target);
  $('predictions').replaceChildren();for(const [name,p]of Object.entries(rv.models)){const box=text('div');box.append(text('h3',name+'：'+(p.matched?'四字段匹配':'四字段不匹配'),p.matched?'good':'bad'),text('pre',pretty(p.prediction)));$('predictions').append(box);}
  if(!Object.keys(rv.models).length)$('predictions').append(text('p','本条没有已有模型预测；页面不会启动推理。','muted'));
  $('context').textContent=pretty(r.context);$('result').textContent=pretty(rv.task_result);$('summaryLink').href=rv.source_summary;$('eventsLink').href=rv.source_events;
  $('raw').textContent=pretty(r);$('stateBrief').textContent='stationary='+r.state.stationary+' · tilt='+Number(r.state.tilt_deg).toFixed(2)+'° · time='+Number(r.state.sim_time_s).toFixed(2)+' s';
  const table=document.createElement('table'),head=document.createElement('tr');for(const v of ['颜色','相机','X（前）','Y（左）','置信度'])head.append(text('th',v));table.append(head);
  for(const d of (r.state.detections_rgbd||[])){const tr=document.createElement('tr');for(const v of [d.color,d.camera,Number(d.center_body_m[0]).toFixed(5),Number(d.center_body_m[1]).toFixed(5),Number(d.confidence??0).toFixed(2)])tr.append(text('td',v));table.append(tr);}
  $('detections').replaceChildren(table);
  $('timeline').replaceChildren();for(const t of tasks.get(taskKey(r))){const b=text('button',(t.context.step??t.source.event_id.replace('event_',''))+': '+skill(t)+(t.target.skill?' '+t.target.seconds+'s':''));b.title=t.review.phase+' / '+t.sample_id;if(t.sample_id===r.sample_id)b.className='active';b.onclick=()=>jump(t);$('timeline').append(b);}
  $('mark').value=marks[r.sample_id]?.verdict||'';$('note').value=marks[r.sample_id]?.note||'';$('markStatus').textContent=marks[r.sample_id]?'本条已有本地标记。':'本条尚未标记。';
 }
 const move=n=>{index=Math.max(0,Math.min(selected.length-1,index+n));render();};
 for(const id of ['mode','split','phase','skill','color'])$(id).onchange=()=>filter();
 let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>filter(),180);};
 $('reset').onclick=()=>{resetFilters();$('mode').value='all';filter();};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('slider').oninput=()=>{index=Number($('slider').value);render();};
 $('task').onclick=()=>jump(current());$('play').onclick=()=>{if(timer){stop();return;}if(!selected.length)return;$('play').textContent='暂停';timer=setInterval(()=>{if(index>=selected.length-1){stop();return;}move(1);},1000);};
 function download(name,value){const u=URL.createObjectURL(new Blob([pretty(value)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
 $('downloadRow').onclick=()=>{const r=current();if(r)download(r.sample_id.replaceAll(':','_')+'_review.json',r);};
 $('saveMark').onclick=()=>{const r=current();if(!r)return;if(!$('mark').value)delete marks[r.sample_id];else marks[r.sample_id]={sample_id:r.sample_id,verdict:$('mark').value,note:$('note').value,updated:new Date().toISOString()};try{localStorage.setItem('microduck-review-20261115',JSON.stringify(marks));$('markStatus').textContent='已保存到本浏览器，未改动训练数据。';}catch(_e){$('markStatus').textContent='浏览器存储不可用，请立即下载全部标记保存。';}};
 $('exportMarks').onclick=()=>download('microduck_manual_review.json',{scope:'Human review notes only; not automatically applied to training data',notes:Object.values(marks)});
 document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);});
 window.reviewDebug={records:records.length,selection:()=>selected.length,current:()=>current()?.sample_id};
 filter(records.find(r=>r.review.split==='val'&&r.review.index===161)?.sample_id);
})();
