// Optional developer dependency: npm install playwright. Tests genuine GPU jobs.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs');const path=require('path');
const root=path.resolve(process.env.STUDIO_ROOT||path.join(__dirname,'..'));
const url=process.env.STUDIO_URL||'http://127.0.0.1:8766';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_BIN||path.join(root,'browser/chrome-linux-arm64/chrome')});
 const output=path.join(root,'reports/validation');fs.mkdirSync(output,{recursive:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1200}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.waitForFunction(()=>document.getElementById('connection-label').textContent.includes('已连接'));
  const records=[];
  while(await page.evaluate(async()=>['loading','running','encoding','evaluating','stopping'].includes((await(await fetch('/api/status')).json()).status)))await page.waitForTimeout(500);
  for(const test of [{mode:'uav',profile:'balanced',video:'UAV/MultiUAV-003.mp4',evaluation:'unavailable'},
                     {mode:'coco',profile:'quality',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
                     {mode:'coco',profile:'balanced',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
                     {mode:'coco',profile:'int8',video:'MOT20/MOT20-01-raw.mp4',evaluation:'completed'},
                     {mode:'coco',profile:'quality',video:'VisDrone/VisDrone-val-uav0000137_00458_v-raw.mp4',evaluation:'completed'}]){
   await page.locator(`.mode[data-mode=${test.mode}]`).click();await page.locator('#profile').selectOption(test.profile);
   const source=path.join(root,'videos',test.video);await page.locator('#source-path').fill(source);await page.locator('#source-path').dispatchEvent('change');
   await page.locator('#output-path').fill(path.join(root,'outputs'));await page.locator('#start-button').click();
   const deadline=Date.now()+360000;
   while(true){
     const probe=await page.evaluate(async()=> (await fetch('/api/status')).json());
     if(['completed','failed'].includes(probe.status))break;
     if(Date.now()>deadline)throw Error('GPU job timed out '+JSON.stringify(probe));
     await page.waitForTimeout(500);
   }
   const result=await page.evaluate(async()=> (await fetch('/api/status')).json());
   if(result.status!=='completed'||result.source_path!==source||result.profile!==test.profile||result.evaluation.status!==test.evaluation)throw Error(JSON.stringify(result));
   if(test.evaluation==='completed'&&result.evaluation.confidence_min!==.001)throw Error('AP reused tracking confidence');
   await page.waitForFunction(()=>document.getElementById('video').readyState>=2,null,{timeout:20000});
   if(await page.locator('#video').evaluate(v=>v.videoWidth===0))throw Error('H264 playback failed');
   await page.locator('#video').evaluate(v=>v.pause());
   await page.screenshot({path:path.join(output,`${test.mode}_${test.profile}_${path.basename(test.video)}.png`),fullPage:true});
   records.push(result);
  }
  await page.locator('#profile').selectOption('balanced');await page.waitForTimeout(1100);
  if(await page.locator('#map50').textContent()!=='—')throw Error('Old profile mAP leaked');
  await page.locator('.mode[data-mode=uav]').click();await page.waitForTimeout(1100);
  if(await page.locator('#map50').textContent()!=='—')throw Error('Old mode mAP leaked');
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile horizontal overflow');
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(output,'browser_delivery.json'),JSON.stringify({root,records,errors,passed:['real UAV FP16','real COCO 1280','real VisDrone 1280','real INT8','H264 playback','AP .001','profile/mode AP reset','mobile']},null,2));
  console.log('DELIVERY REAL GPU BROWSER TESTS PASSED');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
