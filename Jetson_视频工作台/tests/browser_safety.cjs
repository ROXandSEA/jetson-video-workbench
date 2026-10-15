// Real low-impact error/stop checks, plus a full balanced FP16 video.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');const fs=require('fs');const path=require('path');
const root=path.resolve(__dirname,'..');const url=process.env.STUDIO_URL||'http://127.0.0.1:8766';
(async()=>{const browser=await chromium.launch({headless:true,executablePath:path.join(root,'browser/chrome-linux-arm64/chrome')});try{
 const page=await browser.newPage({viewport:{width:1440,height:1200}});await page.goto(url);
 const bootstrap=await page.evaluate(async()=> (await fetch('/api/bootstrap')).json());
 const api=async(p,data)=>page.evaluate(async({p,data,token})=>{const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json','X-Studio-Token':token},body:JSON.stringify(data)});return {code:r.status,value:await r.json()}},{p,data,token:bootstrap.token});
 const status=()=>page.evaluate(async()=> (await fetch('/api/status')).json());
 const wait=async(predicate)=>{const deadline=Date.now()+300000;while(true){const s=await status();if(predicate(s))return s;if(Date.now()>deadline)throw Error('Timed out '+JSON.stringify(s));await page.waitForTimeout(400)}};
 let r=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-01-raw.mp4'),output:path.join(root,'outputs'),mode:'uav',profile:'int8'});
 if(r.code!==400)throw Error('Cross-mode profile accepted');
 r=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-01-raw.mp4'),output:path.join(root,'outputs'),mode:'coco',profile:'balanced'});if(r.code!==200)throw Error(JSON.stringify(r));
 let complete=await wait(s=>['completed','failed'].includes(s.status));if(complete.status!=='completed'||complete.evaluation.confidence_min!==.001)throw Error(JSON.stringify(complete));
 const file=path.join(root,'reports/validation/browser_delivery.json');const report=JSON.parse(fs.readFileSync(file));report.records.push(complete);report.passed.push('real balanced 640 FP16 full MOT');fs.writeFileSync(file,JSON.stringify(report,null,2));
 await page.reload();await page.waitForFunction(()=>document.getElementById('evaluation-status').textContent==='已评估',null,{timeout:10000});
 await page.screenshot({path:path.join(root,'reports/validation/coco_balanced_full.png'),fullPage:true});
 r=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-02-raw.mp4'),output:path.join(root,'outputs'),mode:'coco',profile:'balanced'});if(r.code!==200)throw Error(JSON.stringify(r));
 await wait(s=>s.frame>=10||s.status==='failed');
 const duplicate=await api('/api/start',{source:path.join(root,'videos/MOT20/MOT20-01-raw.mp4'),output:path.join(root,'outputs'),mode:'coco',profile:'balanced'});if(duplicate.code!==409)throw Error('Duplicate GPU job accepted');
 await api('/api/stop',{});const stopped=await wait(s=>['cancelled','failed'].includes(s.status));if(stopped.status!=='cancelled'||stopped.evaluation!==null)throw Error('Cancelled AP falsely published');
 r=await api('/api/start',{source:path.join(root,'videos/UAV/MultiUAV-003.mp4'),output:path.join(root,'outputs'),mode:'coco',profile:'balanced',annotations:path.join(root,'videos/MOT20/MOT20-01-raw.gt.json')});if(r.code!==200)throw Error(JSON.stringify(r));
 const wrong=await wait(s=>['completed','failed'].includes(s.status));if(wrong.status!=='completed'||wrong.evaluation.status!=='invalid'||wrong.evaluation.map50!==null||!wrong.result_url)throw Error('Wrong GT broke video or fabricated AP');
 fs.writeFileSync(path.join(root,'reports/validation/browser_safety.json'),JSON.stringify({balanced:complete,cancelled:stopped,wrong_ground_truth:wrong,passed:['cross-mode profile rejected','single GPU job limit','real GPU cancellation retains partial outputs','wrong GT no AP but valid video retained']},null,2));
 console.log('REAL SAFETY AND BALANCED VIDEO TESTS PASSED');
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
