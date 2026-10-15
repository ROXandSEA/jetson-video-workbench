// Optional dev dependency: playwright. Uses real GPU, encoders and H.264 playback.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs');const path=require('path');
const root=path.resolve(__dirname,'..'),url=process.env.STUDIO_URL||'http://127.0.0.1:8765';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:path.join(root,'browser/chrome-linux-arm64/chrome')});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1200}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
  await page.waitForFunction(()=>document.getElementById('connection-label').textContent.includes('已连接'));
  const bootstrap=await page.evaluate(async()=> (await fetch('/api/bootstrap')).json());
  const api=(p,data)=>page.evaluate(async({p,data,token})=>{
   const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json','X-Studio-Token':token},body:JSON.stringify(data)});
   return {code:r.status,value:await r.json()};},{p,data,token:bootstrap.token});
  const status=()=>page.evaluate(async()=> (await fetch('/api/status')).json());
  const wait=async(predicate)=>{const end=Date.now()+240000;while(true){const s=await status();if(predicate(s))return s;if(Date.now()>end)throw Error(JSON.stringify(s));await page.waitForTimeout(250)}};
  const records=[];
  for(const test of [
   {mode:'uav',profile:'balanced',video:'UAV/MultiUAV-003.mp4',evaluation:'unavailable'},
   {mode:'coco',profile:'balanced',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
   {mode:'coco',profile:'quality',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
   {mode:'coco',profile:'quality',video:'VisDrone/VisDrone-val-uav0000137_00458_v-raw.mp4',evaluation:'completed'},
   {mode:'coco',profile:'fast_motion',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
   {mode:'coco',profile:'fast_static',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
   {mode:'coco',profile:'speed',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
   {mode:'coco',profile:'speed',video:'MOT17/MOT17-09-DPM-raw.mp4',evaluation:'completed'}]){
   await page.locator(`.mode[data-mode=${test.mode}]`).click();await page.locator('#profile').selectOption(test.profile);
   await page.locator('#source-path').fill(path.join(root,'videos',test.video));await page.locator('#source-path').dispatchEvent('change');
   await page.locator('#output-path').fill(path.join(root,'.state/ceiling_browser_outputs'));
   await page.locator('#start-button').click();await wait(s=>['completed','failed'].includes(s.status));
   const state=await status();if(state.status!=='completed'||state.evaluation?.status!==test.evaluation)throw Error(JSON.stringify(state));
   const report=JSON.parse(fs.readFileSync(path.join(state.output_dir,'pipeline.json')));
   if(report.frames!==state.total||!report.encoder.includes('NVENC'))throw Error('Incomplete output or hardware encoder unused');
   await page.waitForFunction(()=>document.getElementById('video').readyState>=2,null,{timeout:15000});
   const video=await page.locator('#video').evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}));
   if(video.width!==report.source_size[0]||video.height!==report.source_size[1])throw Error('Output resolution changed');
   await page.locator('#video').evaluate(v=>v.pause());
   await page.screenshot({path:path.join(root,'reports/validation',`ceiling_${test.mode}_${test.profile}_${path.basename(test.video)}.png`),fullPage:true});
   records.push({state,pipeline:report,playback:video});console.log(test.video,test.profile,report.processing_fps.toFixed(2),'FPS');
  }
  let response=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-05-raw.mp4'),output:path.join(root,'.state/ceiling_browser_outputs'),mode:'coco',profile:'speed'});
  if(response.code!==200)throw Error(JSON.stringify(response));await wait(s=>s.frame>=20||s.status==='failed');
  const duplicate=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-01-raw.mp4'),output:path.join(root,'.state/ceiling_browser_outputs'),mode:'coco',profile:'balanced'});
  if(duplicate.code!==409)throw Error('Concurrent GPU job accepted');await api('/api/stop',{});
  const cancelled=await wait(s=>['cancelled','failed'].includes(s.status));
  if(cancelled.status!=='cancelled'||cancelled.evaluation!==null||!fs.existsSync(path.join(cancelled.output_dir,'result.mp4')))throw Error('Cancellation did not retain partial MP4');
  response=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-01-raw.mp4'),output:path.join(root,'.state/ceiling_browser_outputs'),mode:'coco',profile:'balanced',annotations:path.join(root,'videos/VisDrone/VisDrone-val-uav0000137_00458_v-raw.gt.json')});
  if(response.code!==200)throw Error(JSON.stringify(response));const wrong=await wait(s=>['completed','failed'].includes(s.status));
  if(wrong.status!=='completed'||wrong.evaluation?.status!=='invalid'||!wrong.result_url)throw Error('Wrong GT handling broke');
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(root,'reports/validation/browser_ceiling.json'),JSON.stringify({records,cancelled,wrong_ground_truth:wrong,errors,passed:['full UAV','full COCO640','full COCO1280','full VisDrone1280','NVENC for every mode','original frame count and size','H264 browser playback','cancel keeps partial MP4 without AP','reject concurrent job','wrong GT keeps result without fabricated AP','CUDA Graph fast profiles','MOT17 additional video','cancel CUDA Graph safely']},null,2));
  console.log('CEILING BROWSER TESTS PASSED');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
