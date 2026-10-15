// Developer-only Playwright. Verifies both new profiles and mid-tile cancellation.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..'),url=process.env.STUDIO_URL||'http://127.0.0.1:8765';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:path.join(root,'browser/chrome-linux-arm64/chrome')});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
  await page.waitForFunction(()=>document.getElementById('connection-label').textContent.includes('已连接'));
  const bootstrap=await page.evaluate(async()=>await (await fetch('/api/bootstrap')).json());
  const api=(p,data)=>page.evaluate(async({p,data,token})=>{const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json','X-Studio-Token':token},body:JSON.stringify(data)});return {code:r.status,value:await r.json()}},{p,data,token:bootstrap.token});
  const status=()=>page.evaluate(async()=>await (await fetch('/api/status')).json());
  const wait=async(fn)=>{const end=Date.now()+180000;while(Date.now()<end){const s=await status();if(fn(s))return s;await page.waitForTimeout(100)}throw Error('Timeout')};
  const out=path.join(root,'.state/small_input_browser_outputs');
  let r=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-05-raw.mp4'),output:out,mode:'coco',profile:'speed_detail'});
  if(r.code!==200)throw Error(JSON.stringify(r));await wait(s=>s.frame>=15||s.status==='failed');
  const duplicate=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-01-raw.mp4'),output:out,mode:'coco',profile:'speed'});
  if(duplicate.code!==409)throw Error('Concurrent job accepted');await api('/api/stop',{});
  const cancelled=await wait(s=>['cancelled','failed'].includes(s.status));
  if(cancelled.status!=='cancelled'||cancelled.evaluation!==null||!fs.existsSync(path.join(cancelled.output_dir,'result.mp4')))throw Error('Tile cancellation failed');
  const records=[];
  for(const profile of ['speed','speed_detail']){
   await page.locator('.mode[data-mode=coco]').click();await page.locator('#profile').selectOption(profile);
   await page.locator('#source-path').fill(path.join(root,'videos/MOT20/MOT20-01-raw.mp4'));await page.locator('#source-path').dispatchEvent('change');
   await page.locator('#output-path').fill(out);await page.locator('#start-button').click();
   const state=await wait(s=>['completed','failed'].includes(s.status));
   if(state.status!=='completed'||state.evaluation?.status!=='completed')throw Error(JSON.stringify(state));
   const p=JSON.parse(fs.readFileSync(path.join(state.output_dir,'pipeline.json')));
   if(p.frames!==429||p.detected_frames!==429||p.detector_inferences!==429*(profile==='speed_detail'?5:1)||!p.encoder.includes('NVENC'))throw Error('View/frame/encoder mismatch');
   await page.waitForFunction(()=>document.getElementById('video').readyState>=2,null,{timeout:15000});
   const playback=await page.locator('#video').evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}));
   if(playback.width!==960||playback.height!==540)throw Error('Changed output dimensions');
   await page.locator('#video').evaluate(v=>v.pause());
   await page.screenshot({path:path.join(root,'reports/validation',`small_input_${profile}.png`),fullPage:true});
   records.push({state,pipeline:p,playback});console.log(profile,p.processing_fps.toFixed(2),'FPS',state.evaluation.map50);
  }
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(root,'reports/validation/browser_small_input.json'),JSON.stringify({records,cancelled,errors,passed:['both new profiles','every source frame','five explicit views','NVENC','H264 browser playback','original output dimensions','mid-tile cancellation','reject concurrent GPU task']},null,2));
  console.log('SMALL INPUT BROWSER PASSED');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
