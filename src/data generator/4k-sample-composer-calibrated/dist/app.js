(() => {
  'use strict';
  const W = 3840, H = 2160, MAX_ASSETS = 16;
  const $ = id => document.getElementById(id);
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  const state = { background:null, backgroundName:'', backgroundTransform:{x:W/2,y:H/2,scale:1}, editMode:'object', assets:[], objects:[], selected:null, importedJsonName:'', importedJsonLoaded:false, pendingAnnotations:[], zoom:1, history:[], future:[], interaction:null };
  let toastTimer, wheelHistoryTimer;
  let importing=false, exporting=false, dirty=false;
  const assetArchive=new Map(), backgroundArchive=new Map();
  let sourceData={};
  const metadataIds=['frameValue','poseX','poseY','poseZ','exportName'];
  const clone=value=>JSON.parse(JSON.stringify(value));

  function toast(message, error=false){ const el=$('toast'); el.textContent=message; el.className='toast show'+(error?' error':''); clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.className='toast',2400); }
  function cleanId(name){ return name.replace(/\.[^.]+$/,'').trim().replace(/\s+/g,'_').replace(/[^\w\-\u4e00-\u9fff]/g,'_') || 'object'; }
  function uniqueAssetId(base){ let id=base,n=2; while(state.assets.some(a=>a.id===id)||assetArchive.has(id)) id=`${base}_${n++}`; return id; }
  function fileToImage(file){ return new Promise((resolve,reject)=>{ const url=URL.createObjectURL(file),img=new Image(); img.onload=()=>resolve({img,url}); img.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('图片无法读取'));}; img.src=url; }); }

  async function assetCalibration(file,img){
    const registry=window.SPRITE_BOX_CALIBRATION;
    if(!registry||!Array.isArray(registry.assets))throw new Error('缺少素材校准文件，请完整解压后重新打开');
    if(!registry.assets.length)return null;
    if(!crypto.subtle||typeof file.arrayBuffer!=='function')throw new Error('浏览器不能校验素材，请使用新版 Chrome 或 Edge');
    const digest=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
    const hash=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
    const rule=registry.assets.find(a=>a.sha256===hash);
    if(!rule){
      if(registry.assets.some(a=>a.filename.toLowerCase()===file.name.toLowerCase()))throw new Error(`${file.name} 与已校准原素材不一致，请先重新校准`);
      return null;
    }
    const b=rule.reference_bbox_xyxy;
    if(rule.width!==img.naturalWidth||rule.height!==img.naturalHeight||!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||b[2]<=b[0]||b[3]<=b[1])throw new Error(`${file.name} 的校准信息无效`);
    return {...rule,calibration_id:registry.id};
  }

  async function loadBackground(file){
    if(!file || !/^image\/(png|jpeg|webp)$/.test(file.type)){ toast('请选择 PNG、JPG 或 WEBP 背景图',true); return; }
    try{ const data=await fileToImage(file); if(data.img.naturalWidth!==W||data.img.naturalHeight!==H){ URL.revokeObjectURL(data.url); toast(`背景尺寸为 ${data.img.naturalWidth} × ${data.img.naturalHeight}，必须是 3840 × 2160`,true); return; }
      backgroundArchive.set(data.url,data); state.background=data; state.backgroundName=file.name; state.backgroundTransform={x:W/2,y:H/2,scale:1}; $('emptyCanvas').classList.add('hidden'); $('bgStatus').className='status-pill ready'; $('bgStatus').innerHTML='<i></i> 背景 3840 × 2160'; $('backgroundModeBtn').disabled=false; enableExports(); pushHistory(); syncUI(); render(); fitCanvas(); toast('背景图已载入');
    }catch(e){toast(e.message,true)}
  }

  async function loadAssets(files){
    const list=[...files].filter(f=>f.type==='image/png'||(!f.type&&/\.png$/i.test(f.name))); if(!list.length){toast('请选择 PNG 素材',true);return;}
    let succeeded=0;const errors=[];const room=MAX_ASSETS-state.assets.length; if(room<=0){toast('素材库最多 16 张 PNG',true);return;}
    if(list.length>room) toast(`只导入前 ${room} 张，素材库上限为 16 张`,true);
    for(const file of list.slice(0,room)){let data;try{data=await fileToImage(file);const calibration=await assetCalibration(file,data.img);state.assets.push({id:uniqueAssetId(calibration?.class_name||cleanId(file.name)),name:file.name,img:data.img,url:data.url,w:data.img.naturalWidth,h:data.img.naturalHeight,calibration});succeeded++;}catch(e){if(data)URL.revokeObjectURL(data.url);errors.push(e.message);}}
    state.assets.forEach(a=>assetArchive.set(a.id,a)); updateAssets(); const resolved=resolvePendingAnnotations(false); pushHistory();
    const calibrated=state.assets.filter(a=>a.calibration).length;
    toast(errors.length?errors.join('；'):`已导入 ${succeeded} 张；已校准 ${calibrated}/${state.assets.length}${resolved?`；恢复 ${resolved} 个标注`:''}`,!!errors.length);
  }

  function validateAnnotation(a){return a&&typeof a.object_id==='string'&&a.object_id.trim().length>0&&Array.isArray(a.bbox)&&a.bbox.length===4&&a.bbox.every(Number.isFinite)&&a.bbox[2]>a.bbox[0]&&a.bbox[3]>a.bbox[1];}
  async function loadJson(file){
    const boxesOnly=$('jsonImportMode').value!=='sprites';
    if(!file)return;let data;try{data=JSON.parse((await file.text()).replace(/^\uFEFF/,''));}catch(_){toast('JSON 文件无法解析',true);return;}
    if(!data||!Array.isArray(data.annotations)||!data.annotations.every(validateAnnotation)|| (data.frame!==undefined&&(!Number.isInteger(data.frame)||data.frame<0)) || (data.pose!==undefined&&(!data.pose||typeof data.pose!=='object'||['x','y','z'].some(k=>data.pose[k]!==undefined&&!Number.isFinite(data.pose[k]))))){toast('JSON 格式不符合要求：请检查 annotations 和 bbox',true);return;}
    if(boxesOnly&&data.annotations.some(a=>a.bbox[0]<0||a.bbox[1]<0||a.bbox[2]>W||a.bbox[3]>H)){toast('标注框超出 3840×2160 范围，请检查 JSON 坐标',true);return;}
    if((state.objects.length||state.pendingAnnotations.length)&&!confirm('导入 JSON 会替换当前物体标注，是否继续？'))return;
    sourceData=clone(data);state.objects=[];state.selected=null;state.pendingAnnotations=data.annotations.map((a,sourceIndex)=>({object_id:a.object_id,bbox:a.bbox.map(Number),sourceIndex}));state.importedJsonName=file.name;state.importedJsonLoaded=true;
    if(boxesOnly){state.objects=state.pendingAnnotations.map(a=>boxObject(a));state.pendingAnnotations=[];state.backgroundTransform={x:W/2,y:H/2,scale:1};setMode('object');}
    $('frameValue').value=Number.isFinite(data.frame)?data.frame:0;$('poseX').value=Number(data.pose?.x)||0;$('poseY').value=Number(data.pose?.y)||0;$('poseZ').value=Number.isFinite(Number(data.pose?.z))?Number(data.pose.z):-600;
    $('exportName').value=`${file.name.replace(/\.json$/i,'')}_modified`;const resolved=resolvePendingAnnotations(false);pushHistory();syncUI();render();toast(boxesOnly?`已显示 ${data.annotations.length} 个标注框，无需 PNG`:`已导入 ${data.annotations.length} 个标注${resolved?`，已匹配 ${resolved} 个 PNG`:''}`);
  }
  function resolvePendingAnnotations(record=true){
    if(!state.pendingAnnotations.length){updateImportStatus();return 0;}const unresolved=[];let count=0;
    for(const ann of state.pendingAnnotations){const a=state.assets.find(v=>v.id===ann.object_id||v.name.replace(/\.png$/i,'')===ann.object_id)||state.assets.find(v=>v.calibration&&v.calibration.class_name===ann.object_id.replace(/_\d+$/,''));if(!a){unresolved.push(ann);continue;}const [x1,y1,x2,y2]=ann.bbox,b=a.calibration?.reference_bbox_xyxy||[0,0,a.w,a.h],sx=(x2-x1)/(b[2]-b[0]),sy=(y2-y1)/(b[3]-b[1]);state.objects.push({uid:crypto.randomUUID?.()||`${Date.now()}-${Math.random()}`,asset:a,objectId:ann.object_id,x:(x1+x2)/2-((b[0]+b[2]-a.w)/2)*sx,y:(y1+y2)/2-((b[1]+b[3]-a.h)/2)*sy,scale:sx,scaleY:sy,rotation:0,sourceIndex:ann.sourceIndex});count++;}
    state.pendingAnnotations=unresolved;if(count&&record)pushHistory();syncUI();render();updateImportStatus();return count;
  }
  function updateImportStatus(){const el=$('importStatus');if(!state.importedJsonLoaded){el.className='import-status';el.textContent='未导入原有 JSON';el.title='';return;}const matched=state.objects.filter(o=>Number.isInteger(o.sourceIndex)).length,pending=state.pendingAnnotations.length;el.className=`import-status ${pending?'pending':'ready'}`;el.textContent=pending?`${state.importedJsonName}：已恢复 ${matched} 个，等待 ${pending} 个同名 PNG`:`${state.importedJsonName}：${matched} 个标注已载入${state.objects.some(o=>o.annotationOnly)?'（纯标注框，无需 PNG）':''}`;el.title=pending?`缺少素材：${[...new Set(state.pendingAnnotations.map(a=>a.object_id))].join('、')}`:'';if(pending)el.textContent+='。'+el.title;}

  function updateAssets(){
    $('assetCount').textContent=state.assets.length; $('calibrationStatus').textContent=`已校准 ${state.assets.filter(a=>a.calibration).length}/${state.assets.length} 个素材。未校准素材仍按 PNG 矩形生成框。`; $('assetDrop').classList.toggle('hidden',state.assets.length>0); const grid=$('assetGrid'); grid.innerHTML='';
    state.assets.forEach((a,i)=>{const card=document.createElement('div');card.className='asset-card';card.draggable=true;card.title='点击添加到画布';card.innerHTML=`<div class="asset-thumb"><img src="${a.url}" alt=""></div><span class="asset-name"></span><button class="asset-add" aria-label="添加 ${a.id}">＋</button><button class="asset-remove" aria-label="移除素材 ${a.id}">×</button>`;card.querySelector('.asset-name').textContent=a.id;card.addEventListener('click',e=>{if(!e.target.closest('.asset-remove'))addObject(i)});card.querySelector('.asset-remove').onclick=e=>{e.stopPropagation();removeAsset(i)};card.addEventListener('dragstart',e=>e.dataTransfer.setData('text/asset-index',String(i)));grid.appendChild(card);});
  }
  function removeAsset(i){const a=state.assets[i];if(state.objects.some(o=>o.asset===a)){toast('该素材已在画布中，先删除对应实例',true);return;}state.assets.splice(i,1);updateAssets();pushHistory();}
  function addObject(assetIndex,x=W/2,y=H/2){if(!state.background){toast('请先导入 3840 × 2160 背景图',true);return;}const a=state.assets[assetIndex];if(!a)return;const fit=Math.min(1,Math.min(W*.22/a.w,H*.22/a.h));const o={uid:crypto.randomUUID?.()||`${Date.now()}-${Math.random()}`,asset:a,objectId:a.id,x,y,scale:fit,scaleY:fit,rotation:0};state.objects.push(o);state.selected=o.uid;setMode('object');pushHistory();syncUI();render();toast(`已添加 ${a.id}`);}

  function selected(){return state.objects.find(o=>o.uid===state.selected)||null;}
  function objectCorners(o){const hw=o.asset.w*o.scale/2,hh=o.asset.h*(o.scaleY??o.scale)/2,c=Math.cos(o.rotation),s=Math.sin(o.rotation);return [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]].map(([x,y])=>({x:o.x+x*c-y*s,y:o.y+x*s+y*c}));}
  function referenceCorners(o){
    const b=o.asset.calibration.reference_bbox_xyxy,c=Math.cos(o.rotation),s=Math.sin(o.rotation);
    return [[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]].map(([px,py])=>{
      const x=(px-o.asset.w/2)*o.scale,y=(py-o.asset.h/2)*(o.scaleY??o.scale);
      return {x:o.x+x*c-y*s,y:o.y+x*s+y*c};
    });
  }
  function bbox(o){if(o.annotationOnly)return [o.x-o.asset.w/2,o.y-o.asset.h/2,o.x+o.asset.w/2,o.y+o.asset.h/2];const p=o.asset.calibration?referenceCorners(o):objectCorners(o),xs=p.map(v=>v.x),ys=p.map(v=>v.y),clip=(v,max)=>Math.max(0,Math.min(max,o.asset.calibration?Math.round(v*1e6)/1e6:Math.round(v)));return [clip(Math.min(...xs),W),clip(Math.min(...ys),H),clip(Math.max(...xs),W),clip(Math.max(...ys),H)];}
  function pointInObject(o,x,y){const c=Math.cos(-o.rotation),s=Math.sin(-o.rotation),dx=x-o.x,dy=y-o.y,lx=dx*c-dy*s,ly=dx*s+dy*c;return Math.abs(lx)<=o.asset.w*o.scale/2&&Math.abs(ly)<=o.asset.h*(o.scaleY??o.scale)/2;}
  function hitObject(x,y){for(let i=state.objects.length-1;i>=0;i--)if(pointInObject(state.objects[i],x,y))return state.objects[i];return null;}
  function canvasPoint(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*W/r.width,y:(e.clientY-r.top)*H/r.height,displayScale:r.width/W};}

  function render(exporting=false){ctx.clearRect(0,0,W,H);ctx.fillStyle='#05070a';ctx.fillRect(0,0,W,H);if(state.background){const b=state.backgroundTransform;ctx.save();ctx.translate(b.x,b.y);ctx.scale(b.scale,b.scale);ctx.drawImage(state.background.img,-W/2,-H/2,W,H);ctx.restore();}state.objects.forEach(o=>{if(o.annotationOnly)return;ctx.save();ctx.translate(o.x,o.y);ctx.rotate(o.rotation);ctx.scale(o.scale,o.scaleY??o.scale);ctx.drawImage(o.asset.img,-o.asset.w/2,-o.asset.h/2);ctx.restore();});if(!exporting){state.objects.forEach(drawBoxOverlay);if(state.editMode==='object'){const o=selected();if(o){if(o.annotationOnly)drawBoxHandles(o);else drawSelection(o);}}}}
  function drawSelection(o){const corners=objectCorners(o);ctx.save();ctx.strokeStyle='#36d6c7';ctx.lineWidth=2/(canvas.getBoundingClientRect().width/W);ctx.setLineDash([26/state.zoom,16/state.zoom]);ctx.beginPath();ctx.moveTo(corners[0].x,corners[0].y);corners.slice(1).forEach(p=>ctx.lineTo(p.x,p.y));ctx.closePath();ctx.stroke();ctx.setLineDash([]);const r=5/(canvas.getBoundingClientRect().width/W);corners.forEach(p=>{ctx.fillStyle='#0b1018';ctx.strokeStyle='#36d6c7';ctx.lineWidth=2/(canvas.getBoundingClientRect().width/W);ctx.beginPath();ctx.rect(p.x-r,p.y-r,r*2,r*2);ctx.fill();ctx.stroke();});const top={x:(corners[0].x+corners[1].x)/2,y:(corners[0].y+corners[1].y)/2},off=24/(canvas.getBoundingClientRect().width/W),rot={x:top.x+Math.sin(o.rotation)*off,y:top.y-Math.cos(o.rotation)*off};ctx.beginPath();ctx.moveTo(top.x,top.y);ctx.lineTo(rot.x,rot.y);ctx.stroke();ctx.fillStyle='#ff9f43';ctx.beginPath();ctx.arc(rot.x,rot.y,6/(canvas.getBoundingClientRect().width/W),0,Math.PI*2);ctx.fill();ctx.restore();}
  function handleAt(o,x,y){if(o.annotationOnly){const r=7/(canvas.getBoundingClientRect().width/W);const [l,t,rgt,b]=bbox(o);if(x>l+(rgt-l)*.3&&x<rgt-(rgt-l)*.3&&y>t+(b-t)*.3&&y<b-(b-t)*.3)return null;const handles=boxHandles(o).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y));const h=handles.find(h=>Math.hypot(h.x-x,h.y-y)<r);return h?'box-'+h.key:null;}const r=8/(canvas.getBoundingClientRect().width/W),corners=objectCorners(o);for(const p of corners)if(Math.hypot(p.x-x,p.y-y)<r)return'resize';const top={x:(corners[0].x+corners[1].x)/2,y:(corners[0].y+corners[1].y)/2},off=24/(canvas.getBoundingClientRect().width/W),rot={x:top.x+Math.sin(o.rotation)*off,y:top.y-Math.cos(o.rotation)*off};if(Math.hypot(rot.x-x,rot.y-y)<r*1.2)return'rotate';return null;}

  canvas.addEventListener('pointerdown',e=>{if(!state.background)return;if(e.button!==0||state.interaction)return;const p=canvasPoint(e);if(state.editMode==='background'){const b=state.backgroundTransform;state.interaction={type:'background-move',start:p,ox:b.x,oy:b.y};canvas.setPointerCapture(e.pointerId);return;}const cur=selected(),handle=cur&&handleAt(cur,p.x,p.y);let o=cur;if(!handle){o=hitObject(p.x,p.y);state.selected=o?.uid||null;}if(o){state.interaction={type:handle||'move',uid:o.uid,start:p,ox:o.x,oy:o.y,bounds:o.annotationOnly?bbox(o):null,os:o.scale,osy:o.scaleY??o.scale,or:o.rotation,dist:Math.hypot(p.x-o.x,p.y-o.y),angle:Math.atan2(p.y-o.y,p.x-o.x)};canvas.setPointerCapture(e.pointerId);}syncUI();render();});
  canvas.addEventListener('pointermove',e=>{const p=canvasPoint(e);$('pointerCoords').textContent=`${Math.round(p.x)}, ${Math.round(p.y)}`;const it=state.interaction;if(!it)return;if(it.type==='background-move'){state.backgroundTransform.x=it.ox+p.x-it.start.x;state.backgroundTransform.y=it.oy+p.y-it.start.y;syncBackgroundValues();render();return;}const o=state.objects.find(v=>v.uid===it.uid);if(!o)return;if(it.type.startsWith('box-')){resizeBox(o,it.bounds,it.type.slice(4),p.x,p.y);}else if(it.type==='move'){o.x=it.ox+p.x-it.start.x;o.y=it.oy+p.y-it.start.y;if(o.annotationOnly)clampBox(o);}else if(it.type==='resize'){const factor=Math.hypot(p.x-o.x,p.y-o.y)/Math.max(1,it.dist);const safe=Math.max(.0001/Math.min(it.os,it.osy),Math.min(100/Math.max(it.os,it.osy),factor));o.scale=it.os*safe;o.scaleY=it.osy*safe;}else{o.rotation=it.or+Math.atan2(p.y-o.y,p.x-o.x)-it.angle;}syncInspectorValues();render();});
  canvas.addEventListener('pointerup',e=>{if(state.interaction){state.interaction=null;pushHistory();canvas.releasePointerCapture(e.pointerId);}});canvas.addEventListener('pointerleave',()=>{if(!state.interaction)$('pointerCoords').textContent='—, —';});
  canvas.addEventListener('dragover',e=>{e.preventDefault();canvas.style.outline='2px solid #36d6c7';});canvas.addEventListener('dragleave',()=>canvas.style.outline='');canvas.addEventListener('drop',e=>{e.preventDefault();canvas.style.outline='';const raw=e.dataTransfer.getData('text/asset-index'),i=raw===''?NaN:Number(raw);const p=canvasPoint(e);if(Number.isInteger(i))addObject(i,p.x,p.y);else if(e.dataTransfer.files[0]){const file=e.dataTransfer.files[0];/\.json$/i.test(file.name)?loadJson(file):loadBackground(file);}});
  canvas.addEventListener('wheel',e=>{if(state.editMode!=='background'||!state.background)return;e.preventDefault();const p=canvasPoint(e),b=state.backgroundTransform,old=b.scale,next=Math.max(.1,Math.min(5,old*Math.exp(-e.deltaY*.001)));b.x=p.x-(p.x-b.x)*(next/old);b.y=p.y-(p.y-b.y)*(next/old);b.scale=next;syncBackgroundValues();render();clearTimeout(wheelHistoryTimer);wheelHistoryTimer=setTimeout(pushHistory,180);},{passive:false});
  $('emptyCanvas').addEventListener('dragover',e=>{e.preventDefault();$('emptyCanvas').classList.add('drop-zone','drag')});$('emptyCanvas').addEventListener('dragleave',()=>$('emptyCanvas').classList.remove('drag'));$('emptyCanvas').addEventListener('drop',e=>{e.preventDefault();$('emptyCanvas').classList.remove('drag');loadBackground(e.dataTransfer.files[0]);});

  function setMode(mode){if(mode==='background'&&(!state.background||state.objects.some(o=>o.annotationOnly)))return;state.editMode=mode;flushInteraction();$('objectModeBtn').classList.toggle('active',mode==='object');$('backgroundModeBtn').classList.toggle('active',mode==='background');$('canvasShell').classList.toggle('background-mode',mode==='background');syncUI();render();}
  function syncUI(){$('backgroundModeBtn').disabled=!state.background||state.objects.some(o=>o.annotationOnly);const o=selected(),bgMode=state.editMode==='background';$('backgroundInspector').classList.toggle('hidden',!bgMode);$('emptyInspector').classList.toggle('hidden',bgMode||!!o);$('inspector').classList.toggle('hidden',bgMode||!o);$('inspectorTitle').textContent=bgMode?'背景属性':o?.annotationOnly?'标注框属性':'物体属性';$('selectionHint').textContent=bgMode?'移动和缩放背景':(o?o.objectId:'未选择物体');$('objectStatus').textContent=state.pendingAnnotations.length?`${state.objects.length} 个实例 · ${state.pendingAnnotations.length} 个待匹配`:`${state.objects.length} 个实例`;if(bgMode)syncBackgroundValues();else if(o)syncInspectorValues();updateImportStatus();const list=$('objectList');list.replaceChildren();const empty=document.createElement('option');empty.value='';empty.textContent='选择实例（含画布外物体）';list.appendChild(empty);state.objects.forEach((item,i)=>{const option=document.createElement('option');option.value=item.uid;option.textContent=(i+1)+'. '+item.objectId+(item.annotationOnly?' [框]':' [PNG]');list.appendChild(option);});list.value=state.selected||'';enableExports();}
  function syncInspectorValues(){const o=selected();if(!o)return;$('boxFields').classList.toggle('hidden',!o.annotationOnly);$('scale').closest('label').classList.toggle('hidden',!!o.annotationOnly);$('rotation').closest('label').classList.toggle('hidden',!!o.annotationOnly);if(o.annotationOnly)bbox(o).forEach((v,i)=>$('box'+i).value=v);$('objectId').value=o.objectId;$('posX').value=Math.round(o.x);$('posY').value=Math.round(o.y);$('scale').value=Math.round(o.scale*100);$('scaleOut').textContent=`${Math.round(o.scale*100)}%`;$('rotation').value=Math.round(o.rotation*180/Math.PI);$('rotationOut').textContent=`${Math.round(o.rotation*180/Math.PI)}°`;$('bboxValue').textContent=JSON.stringify(bbox(o));}
  function syncBackgroundValues(){const b=state.backgroundTransform;$('bgPosX').value=Math.round(b.x);$('bgPosY').value=Math.round(b.y);$('bgScale').value=Math.round(b.scale*100);$('bgScaleOut').textContent=`${Math.round(b.scale*100)}%`;}
  function bindValue(id,fn,event='change'){$(id).addEventListener(event,()=>{const o=selected();if(!o)return;const field=$(id);if(field.type==='number'&&(field.value.trim()===''||!Number.isFinite(Number(field.value)))){toast('请输入有效数字',true);syncUI();return;}fn(o,field.value);syncUI();render();if(event==='change')pushHistory();});}
  bindValue('objectId',(o,v)=>o.objectId=v.trim()||o.objectId);bindValue('posX',(o,v)=>{o.x=Number(v);if(o.annotationOnly)clampBox(o)});bindValue('posY',(o,v)=>{o.y=Number(v);if(o.annotationOnly)clampBox(o)});bindValue('scale',(o,v)=>{const next=Number(v)/100,old=o.scale;o.scale=next;o.scaleY=(o.scaleY??old)*(next/Math.max(.001,old));},'input');bindValue('rotation',(o,v)=>o.rotation=Number(v)*Math.PI/180,'input');$('scale').addEventListener('change',pushHistory);$('rotation').addEventListener('change',pushHistory);
  $('objectModeBtn').onclick=()=>setMode('object');$('backgroundModeBtn').onclick=()=>setMode('background');
  function setBackgroundScale(next,anchor={x:W/2,y:H/2}){const b=state.backgroundTransform,old=b.scale;b.scale=Math.max(.1,Math.min(5,next));b.x=anchor.x-(anchor.x-b.x)*(b.scale/old);b.y=anchor.y-(anchor.y-b.y)*(b.scale/old);syncBackgroundValues();render();}
  for(const [id,key] of [['bgPosX','x'],['bgPosY','y']])$(id).onchange=()=>{const raw=$(id).value;if(raw.trim()===''||!Number.isFinite(Number(raw))){toast('请输入有效坐标',true);syncBackgroundValues();return;}state.backgroundTransform[key]=Number(raw);pushHistory();render();};$('bgScale').oninput=()=>setBackgroundScale(Number($('bgScale').value)/100);$('bgScale').onchange=pushHistory;$('bgZoomOutBtn').onclick=()=>{setBackgroundScale(state.backgroundTransform.scale/1.1);pushHistory();};$('bgZoomInBtn').onclick=()=>{setBackgroundScale(state.backgroundTransform.scale*1.1);pushHistory();};$('resetBackgroundBtn').onclick=()=>{state.backgroundTransform={x:W/2,y:H/2,scale:1};syncBackgroundValues();pushHistory();render();toast('背景已重置');};
  function deleteSelected(){const i=state.objects.findIndex(o=>o.uid===state.selected);if(i<0)return;state.objects.splice(i,1);state.selected=null;pushHistory();syncUI();render();}
  function duplicate(){const o=selected();if(!o)return;const n={...o,uid:crypto.randomUUID?.()||`${Date.now()}-${Math.random()}`,asset:o.annotationOnly?{...o.asset}:o.asset,x:o.x+35,y:o.y+35,sourceIndex:undefined};if(n.annotationOnly)clampBox(n);state.objects.push(n);state.selected=n.uid;pushHistory();syncUI();render();}
  $('deleteBtn').onclick=deleteSelected;$('duplicateBtn').onclick=duplicate;$('forwardBtn').onclick=()=>{const o=selected();if(!o)return;state.objects=state.objects.filter(v=>v!==o);state.objects.push(o);pushHistory();render();};$('backBtn').onclick=()=>{const o=selected();if(!o)return;state.objects=state.objects.filter(v=>v!==o);state.objects.unshift(o);pushHistory();render();};

  function snapshot(){return{objects:state.objects.map(o=>({uid:o.uid,assetId:o.asset.id,objectId:o.objectId,x:o.x,y:o.y,scale:o.scale,scaleY:o.scaleY??o.scale,rotation:o.rotation,sourceIndex:o.sourceIndex,annotationOnly:!!o.annotationOnly,boxSize:o.annotationOnly?{w:o.asset.w,h:o.asset.h}:null})),assetIds:state.assets.map(a=>a.id),pendingAnnotations:clone(state.pendingAnnotations),backgroundTransform:{...state.backgroundTransform},backgroundUrl:state.background?.url||null,backgroundName:state.backgroundName,sourceData:clone(sourceData),importedJsonName:state.importedJsonName,importedJsonLoaded:state.importedJsonLoaded,metadata:Object.fromEntries(metadataIds.map(id=>[id,$(id).value]))};}
  function restore(snap){state.assets=(snap.assetIds||[]).map(id=>assetArchive.get(id)).filter(Boolean);state.objects=(snap.objects||[]).map(v=>({...v,asset:v.annotationOnly?{id:'box',...v.boxSize}:assetArchive.get(v.assetId)})).filter(o=>o.asset);state.pendingAnnotations=clone(snap.pendingAnnotations||[]);state.backgroundTransform={...snap.backgroundTransform};state.background=backgroundArchive.get(snap.backgroundUrl)||null;state.backgroundName=snap.backgroundName||'';sourceData=clone(snap.sourceData||{});state.importedJsonName=snap.importedJsonName||'';state.importedJsonLoaded=!!snap.importedJsonLoaded;metadataIds.forEach(id=>{if(snap.metadata)$(id).value=snap.metadata[id]});$('emptyCanvas').classList.toggle('hidden',!!state.background);$('backgroundModeBtn').disabled=!state.background;$('bgStatus').className='status-pill '+(state.background?'ready':'waiting');$('bgStatus').textContent=state.background?'背景 3840 × 2160':'等待背景图';state.selected=null;updateAssets();setMode('object');syncUI();render();dirty=true;}
  function pushHistory(){clearTimeout(wheelHistoryTimer);wheelHistoryTimer=null;const s=JSON.stringify(snapshot());if(state.history[state.history.length-1]!==s){dirty=true;state.history.push(s);if(state.history.length>50)state.history.shift();state.future=[];}updateHistoryButtons();}
  function undo(){flushInteraction();if(state.history.length<2)return;state.future.push(state.history.pop());restore(JSON.parse(state.history[state.history.length-1]));updateHistoryButtons();}
  function redo(){flushInteraction();if(!state.future.length)return;const s=state.future.pop();state.history.push(s);restore(JSON.parse(s));updateHistoryButtons();}
  function updateHistoryButtons(){$('undoBtn').disabled=state.history.length<2;$('redoBtn').disabled=!state.future.length;}
  $('undoBtn').onclick=undo;$('redoBtn').onclick=redo;

  function annotationData(){const counts=Object.create(null);const ordered=[...state.objects.map(o=>({object_id:o.objectId,bbox:bbox(o),sourceIndex:o.sourceIndex})).filter(a=>a.bbox[2]>a.bbox[0]&&a.bbox[3]>a.bbox[1]),...state.pendingAnnotations.map(a=>({object_id:a.object_id,bbox:[...a.bbox],sourceIndex:a.sourceIndex}))].sort((a,b)=>(a.sourceIndex??Infinity)-(b.sourceIndex??Infinity));const annotations=ordered.map(({object_id,bbox,sourceIndex})=>({...clone(sourceData.annotations?.[sourceIndex]||{}),object_id,bbox}));annotations.forEach(a=>counts[a.object_id]=(counts[a.object_id]||0)+1);return{...clone(sourceData),frame:Number($('frameValue').value)||0,pose:{...clone(sourceData.pose||{}),x:Number($('poseX').value)||0,y:Number($('poseY').value)||0,z:Number($('poseZ').value)||0},annotations,object_counts:Object.fromEntries(Object.entries(counts).sort(([a],[b])=>a.localeCompare(b)))};}
  function filename(){let name=$('exportName').value.trim().replace(/[<>:"/\\|?*\u0000-\u001f]/g,'_').replace(/[. ]+$/g,'')||'sample_0000';if(name===state.importedJsonName.replace(/\.json$/i,''))name+='_modified';return name;}
  function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1500);}
  async function imageBlob(){const output=document.createElement('canvas');output.width=W;output.height=H;const out=output.getContext('2d');render(true);out.drawImage(canvas,0,0);render();return new Promise((resolve,reject)=>output.toBlob(blob=>blob?resolve(blob):reject(new Error('PNG 编码失败，请重试')),'image/png'));}
  function jsonBlob(){return new Blob([JSON.stringify(annotationData(),null,2)],{type:'application/json'});}
  $('pngBtn').onclick=()=>runExport('png');
  $('jsonBtn').onclick=()=>runExport('json');

  const crcTable=(()=>{const t=new Uint32Array(256);for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;t[n]=c>>>0;}return t;})();
  function crc32(data){let c=0xffffffff;for(const b of data)c=crcTable[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0;}
  function u16(n){return new Uint8Array([n&255,(n>>>8)&255])}function u32(n){return new Uint8Array([n&255,(n>>>8)&255,(n>>>16)&255,(n>>>24)&255])}
  function concat(parts){const len=parts.reduce((n,p)=>n+p.length,0),out=new Uint8Array(len);let at=0;for(const p of parts){out.set(p,at);at+=p.length;}return out;}
  function makeZip(files){const enc=new TextEncoder(),locals=[],centrals=[];let offset=0;for(const f of files){const name=enc.encode(f.name),data=f.data,crc=crc32(data);const local=concat([u32(0x04034b50),u16(20),u16(0x800),u16(0),u16(0),u16(0),u32(crc),u32(data.length),u32(data.length),u16(name.length),u16(0),name,data]);locals.push(local);centrals.push(concat([u32(0x02014b50),u16(20),u16(20),u16(0x800),u16(0),u16(0),u16(0),u32(crc),u32(data.length),u32(data.length),u16(name.length),u16(0),u16(0),u16(0),u16(0),u32(0),u32(offset),name]));offset+=local.length;}const central=concat(centrals),end=concat([u32(0x06054b50),u16(0),u16(0),u16(files.length),u16(files.length),u32(central.length),u32(offset),u16(0)]);return new Blob([...locals,central,end],{type:'application/zip'});}
  async function exportZip(){return runExport('zip');}
  async function runExport(kind){
    if(exporting||importing)return;
    if(kind!=='json'&&(!state.background||state.pendingAnnotations.length)){toast('请先导入背景和所有缺失 PNG；仍可单独导出 JSON',true);return;}
    if(!validateMetadata())return;
    flushInteraction();const name=filename(),json=JSON.stringify(annotationData(),null,2);exporting=true;enableExports();
    try{if(kind==='json'){download(new Blob([json],{type:'application/json'}),name+'.json');}
      else {const png=await imageBlob();if(kind==='png')download(png,name+'.png');else download(makeZip([{name:name+'.png',data:new Uint8Array(await png.arrayBuffer())},{name:name+'.json',data:new TextEncoder().encode(json)}]),name+'.zip');}
      toast('已生成新文件。完整编辑状态仍只在当前页面中，请勿直接刷新。');
    }catch(e){toast('导出失败：'+e.message,true);}finally{exporting=false;enableExports();}
  }
  $('zipBtn').onclick=exportZip;$('zipBtnSide').onclick=exportZip;
  function enableExports(){const locked=importing||exporting,imageOk=!!state.background&&!state.pendingAnnotations.length,jsonOk=!!state.background||state.importedJsonLoaded;$('pngBtn').disabled=locked||!imageOk;$('jsonBtn').disabled=locked||!jsonOk;$('zipBtn').disabled=locked||!imageOk;$('zipBtnSide').disabled=locked||!imageOk;$('exportWarning').textContent=state.pendingAnnotations.length?'缺少 PNG：已暂停图片/ZIP 导出，JSON 导出会保留未匹配标注。':'标注框与标签仅用于预览，样本 PNG 不含框；JSON 保存修改后的 bbox。';}

  function fitCanvas(){state.zoom=1;const wrap=$('stageScroll'),shell=$('canvasShell'),availW=wrap.clientWidth-44,availH=wrap.clientHeight-44;const natural=Math.min(availW/W,availH/H);shell.style.width=`${Math.max(320,W*natural)}px`;$('zoomValue').textContent='视图：适应';render();}
  function setZoom(mult){const shell=$('canvasShell'),base=shell.getBoundingClientRect().width,old=state.zoom;state.zoom=Math.max(.25,Math.min(4,state.zoom*mult));shell.style.width=`${base*state.zoom/old}px`;$('zoomValue').textContent=`视图：${Math.round(state.zoom*100)}%`;render();}
  $('zoomIn').onclick=()=>setZoom(1.2);$('zoomOut').onclick=()=>setZoom(1/1.2);$('fitBtn').onclick=fitCanvas;window.addEventListener('resize',()=>{if($('zoomValue').textContent==='视图：适应')fitCanvas()});
  function clearAll(){if((state.objects.length||state.pendingAnnotations.length)&&!confirm('确定清空画布中的所有物体和待匹配标注吗？'))return;state.objects=[];state.pendingAnnotations=[];state.selected=null;pushHistory();syncUI();render();toast('画布已清空');}
  $('newBtn').onclick=clearAll;
  ['bgInput'].forEach(id=>$(id).onchange=e=>{loadBackground(e.target.files[0]);e.target.value='';});['assetInput','assetInputEmpty'].forEach(id=>$(id).onchange=e=>{loadAssets(e.target.files);e.target.value='';});$('jsonInput').onchange=e=>{loadJson(e.target.files[0]);e.target.value='';};
  $('assetDrop').addEventListener('dragover',e=>{e.preventDefault();$('assetDrop').classList.add('drag')});$('assetDrop').addEventListener('dragleave',()=>$('assetDrop').classList.remove('drag'));$('assetDrop').addEventListener('drop',e=>{e.preventDefault();$('assetDrop').classList.remove('drag');loadAssets(e.dataTransfer.files)});
  document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;const mod=e.ctrlKey||e.metaKey;if(mod&&e.key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?redo():undo();return;}if(mod&&e.key.toLowerCase()==='y'){e.preventDefault();redo();return;}if(mod&&e.key.toLowerCase()==='d'&&state.editMode==='object'){e.preventDefault();duplicate();return;}const d=e.shiftKey?10:1;if(state.editMode==='background'){const b=state.backgroundTransform;if(e.key==='ArrowLeft')b.x-=d;else if(e.key==='ArrowRight')b.x+=d;else if(e.key==='ArrowUp')b.y-=d;else if(e.key==='ArrowDown')b.y+=d;else return;e.preventDefault();pushHistory();syncBackgroundValues();render();return;}const o=selected();if(!o)return;if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();deleteSelected();return;}if(e.key==='ArrowLeft')o.x-=d;else if(e.key==='ArrowRight')o.x+=d;else if(e.key==='ArrowUp')o.y-=d;else if(e.key==='ArrowDown')o.y+=d;else return;e.preventDefault();if(o.annotationOnly)clampBox(o);pushHistory();syncInspectorValues();render();});
  function registerWebMCP(){
    const context=document.modelContext;if(!context?.registerTool)return;const annotations={untrustedContentHint:false};
    const register=tool=>{try{void Promise.resolve(context.registerTool(tool)).catch(()=>{});}catch(_){}}
    register({name:'read_composition',title:'读取当前合成',description:'读取当前画布的元数据、背景变换、实例数量和可见物体标注，不修改画布。',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{...annotations,readOnlyHint:true},execute:()=>({background_loaded:!!state.background,background_transform:{...state.backgroundTransform},canvas:{width:W,height:H},asset_count:state.assets.length,instance_count:state.objects.length,...annotationData()})});
    register({name:'add_asset_instance',title:'添加素材实例',description:'按素材 object_id 在 4K 画布指定中心坐标添加一个实例。背景图和对应 PNG 素材必须已经由用户导入。',inputSchema:{type:'object',properties:{object_id:{type:'string'},x:{type:'number',minimum:0,maximum:W},y:{type:'number',minimum:0,maximum:H}},required:['object_id','x','y'],additionalProperties:false},annotations:{...annotations,readOnlyHint:false},execute:input=>{const i=state.assets.findIndex(a=>a.id===input?.object_id);if(!input||!Number.isFinite(input.x)||!Number.isFinite(input.y)||input.x<0||input.x>W||input.y<0||input.y>H)throw new Error('坐标无效');if(!state.background)throw new Error('请先导入背景图');if(i<0)throw new Error('找不到该 object_id 的素材');addObject(i,input.x,input.y);return{status:'added',object_id:state.assets[i].id,x:input.x,y:input.y,instance_count:state.objects.length};}});
  }

  function flushInteraction(){if(state.interaction||wheelHistoryTimer){state.interaction=null;pushHistory();}}
  canvas.addEventListener('pointercancel',()=>{state.interaction=null;pushHistory();render();});
  function validateMetadata(){for(const id of metadataIds.filter(id=>id!=='exportName')){const el=$(id),v=Number(el.value);if(el.value.trim()===''||!Number.isFinite(v)||(id==='frameValue'&&(!Number.isInteger(v)||v<0))){toast('frame 必须为非负整数；pose 必须为有效数字',true);el.focus();return false;}}return true;}
  metadataIds.forEach(id=>$(id).addEventListener('change',()=>{if(validateMetadata())pushHistory();}));
  function guardImport(fn){return async(...args)=>{if(importing||exporting){toast('正在处理文件，请稍候');return;}flushInteraction();importing=true;enableExports();try{await fn(...args);}catch(e){toast('导入失败：'+e.message,true);}finally{importing=false;enableExports();}};}
  loadBackground=guardImport(loadBackground);loadAssets=guardImport(loadAssets);loadJson=guardImport(loadJson);
  window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
  $('objectList').addEventListener('change',e=>{state.selected=e.target.value||null;setMode('object');});
  $('centerObjectBtn').onclick=()=>{const o=selected();if(!o)return;o.x=W/2;o.y=H/2;pushHistory();syncUI();render();};
  $('changeBackgroundBtn').onclick=()=>$('bgInput').click();


  function boxObject(a){
    const [x1,y1,x2,y2]=a.bbox;
    return {uid:crypto.randomUUID?.()||String(Math.random()),annotationOnly:true,asset:{id:'box',w:x2-x1,h:y2-y1},objectId:a.object_id,x:(x1+x2)/2,y:(y1+y2)/2,scale:1,scaleY:1,rotation:0,sourceIndex:a.sourceIndex};
  }
  function clampBox(o){o.x=Math.max(o.asset.w/2,Math.min(W-o.asset.w/2,o.x));o.y=Math.max(o.asset.h/2,Math.min(H-o.asset.h/2,o.y));}
  function setBoxBounds(o,b){o.asset={...o.asset,w:b[2]-b[0],h:b[3]-b[1]};o.x=(b[0]+b[2])/2;o.y=(b[1]+b[3])/2;}
  function boxHandles(o){const [l,t,r,b]=bbox(o),mx=(l+r)/2,my=(t+b)/2;return [{key:'nw',x:l,y:t},{key:'ne',x:r,y:t},{key:'se',x:r,y:b},{key:'sw',x:l,y:b},{key:'n',x:mx,y:t},{key:'e',x:r,y:my},{key:'s',x:mx,y:b},{key:'w',x:l,y:my}];}
  function resizeBox(o,start,key,x,y){let [l,t,r,b]=start;x=Math.max(0,Math.min(W,Math.round(x)));y=Math.max(0,Math.min(H,Math.round(y)));
    if(key.includes('w'))l=Math.min(r-1,x);if(key.includes('e'))r=Math.max(l+1,x);
    if(key.includes('n'))t=Math.min(b-1,y);if(key.includes('s'))b=Math.max(t+1,y);
    setBoxBounds(o,[l,t,r,b]);
  }
  function drawBoxOverlay(o){const [l,t,r,b]=bbox(o),px=W/canvas.getBoundingClientRect().width;ctx.save();ctx.strokeStyle=o.uid===state.selected?'#ffb454':'#36d6c7';ctx.lineWidth=2*px;ctx.strokeRect(l,t,r-l,b-t);ctx.font=(13*px)+'px sans-serif';ctx.textBaseline='top';
    const label=o.objectId,tw=ctx.measureText(label).width+8*px,tx=Math.max(0,Math.min(W-tw,l)),ty=Math.max(0,t-20*px);
    ctx.fillStyle='#07111eee';ctx.fillRect(tx,ty,tw,19*px);ctx.fillStyle=o.uid===state.selected?'#ffb454':'#8ffff0';ctx.fillText(label,tx+4*px,ty+2*px);ctx.restore();}
  function drawBoxHandles(o){const px=W/canvas.getBoundingClientRect().width;ctx.save();ctx.fillStyle='#ffb454';ctx.strokeStyle='#07111e';ctx.lineWidth=px;for(const h of boxHandles(o)){ctx.fillRect(h.x-4*px,h.y-4*px,8*px,8*px);ctx.strokeRect(h.x-4*px,h.y-4*px,8*px,8*px);}ctx.restore();}
  for(let i=0;i<4;i++)$('box'+i).onchange=()=>{const o=selected();if(!o?.annotationOnly)return;const b=[0,1,2,3].map(i=>Number($('box'+i).value));
    if([0,1,2,3].some(i=>$('box'+i).value.trim()==='')||!b.every(Number.isFinite)||b[0]<0||b[1]<0||b[2]>W||b[3]>H||b[2]<=b[0]||b[3]<=b[1]){toast('bbox 必须在画布内，且 x2 > x1、y2 > y1',true);syncInspectorValues();return;}
    setBoxBounds(o,b);pushHistory();syncInspectorValues();render();};

  pushHistory();dirty=false;syncUI();render();requestAnimationFrame(fitCanvas);registerWebMCP();
})();
