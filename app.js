(()=>{'use strict';
const I=window.CoinInference, geom=I.geometry();
const el=id=>document.getElementById(id), canvas=el('board'),ctx=canvas.getContext('2d');
const select=el('checkpoint');const manifestUrl='./models/manifest.json';
let manifest=null, checkpoint='20000',humanFirst=false,worker=null,seq=0,pending=new Map(),busy=false;
let state,history=[]; const FRAME=720,scale=43,center=360;
function point(a){const [x,y]=geom.coords[a];return [center+x*scale,center+y*scale];}
function role(side){return side===(humanFirst?0:1)?'Player':'AI';}
function makeState(){return {coins:[[],[]],turn:0,last:-1};}
function clone(s){return {coins:[s.coins[0].slice(),s.coins[1].slice()],turn:s.turn,last:s.last};}
function legal(){return I.legal(geom.board,geom.coords,[...state.coins[0],...state.coins[1]],geom.radius);}
function countLegal(){return legal().reduce((x,y)=>x+y,0);}
function gameOver(){return countLegal()===0;}
function setStatus(t,err=false){el('status').textContent=t;el('status').classList.toggle('error',err);}
function updateRoles(){el('orangeRole').textContent=role(0)+' · first';el('greenRole').textContent=role(1)+' · second';for(const [id,on] of [['aiFirst',!humanFirst],['humanFirst',humanFirst]]){el(id).classList.toggle('active',on);el(id).setAttribute('aria-pressed',String(on));}}
function draw(){
 const W=FRAME; ctx.clearRect(0,0,W,W);ctx.fillStyle='#f7fafd';ctx.fillRect(0,0,W,W);
 ctx.beginPath();ctx.arc(center,center,geom.outer*scale,0,2*Math.PI);ctx.fillStyle='#e8f0f7';ctx.fill();ctx.lineWidth=3;ctx.strokeStyle='#8da3b9';ctx.stroke();
 const legalMask=legal();
 if(el('showGrid').checked){for(let i=0;i<I.PIX;i++)if(geom.board[i]){const [x,y]=point(i);ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fillStyle='#9baebf';ctx.fill();}}
 if(el('showLegal').checked){for(let i=0;i<I.PIX;i++)if(legalMask[i]){const [x,y]=point(i);ctx.beginPath();ctx.arc(x,y,9,0,2*Math.PI);ctx.fillStyle='rgba(37,122,195,.28)';ctx.fill();ctx.strokeStyle='rgba(37,122,195,.55)';ctx.lineWidth=1;ctx.stroke();}}
 for(let side=0;side<2;side++)for(let ix=0;ix<state.coins[side].length;ix++){
   const a=state.coins[side][ix];const [x,y]=point(a);ctx.beginPath();ctx.arc(x,y,geom.radius*scale,0,2*Math.PI);ctx.fillStyle=side===0?'#f28c28':'#4caf50';ctx.fill();ctx.lineWidth=3;ctx.strokeStyle='#ffffff';ctx.stroke();ctx.beginPath();ctx.arc(x-14,y-17,geom.radius*scale*.24,0,2*Math.PI);ctx.fillStyle='rgba(255,255,255,.16)';ctx.fill();
 }
 if(el('showCoordinates').checked){ctx.fillStyle='#47647b';ctx.font='13px system-ui';ctx.textAlign='center';for(let k=-6;k<=6;k+=2){ctx.fillText(String(k),center+k*scale,center+geom.outer*scale+24);if(k!==0)ctx.fillText(String(-k),center-geom.outer*scale-19,center+k*scale+5);}}
 el('coinCount').textContent=(state.coins[0].length+state.coins[1].length)+' coins played';el('movesCount').textContent=legalMask.reduce((a,b)=>a+b,0)+' legal positions';
 el('undoHuman').disabled=busy||history.length===0;
}
function put(a){state.coins[state.turn].push(a);state.last=a;state.turn=1-state.turn;draw();}
function showTurn(){if(gameOver()){const winner=role(1-state.turn);setStatus(`${winner} wins! There are no legal coin placements left.`);return;}setStatus(state.turn===(humanFirst?0:1)?'Your turn — click a highlighted point.':'AI is choosing a move…');}
function send(data){return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});worker.postMessage({id,...data});});}
async function loadSelected(){busy=true;el('loading').textContent='Loading model…';draw();const key=checkpoint;try{await send({type:'load',checkpoint:key,meta:manifest.checkpoints[key].tensor_offsets});if(key!==checkpoint)return;el('loading').textContent='Model ready';busy=false;draw();if(state.turn!==(humanFirst?0:1)&&!gameOver())await aiMove();else showTurn();}catch(e){busy=false;el('loading').textContent='Loading failed';setStatus('Model loading failed: '+e.message,true);}}
async function aiMove(){if(busy||gameOver()||state.turn===(humanFirst?0:1))return;busy=true;setStatus('AI is calculating its move…');draw();const myState=clone(state),key=checkpoint;try{const result=await send({type:'predict',checkpoint:key,state:myState,radius:geom.radius,outer:geom.outer,hole:geom.hole});if(key!==checkpoint||state!==myState&&JSON.stringify(state)!==JSON.stringify(myState))return;const a=result.action;if(a<0||!legal()[a])throw new Error('Model returned an illegal move');put(a);showTurn();}catch(e){setStatus('Could not calculate the AI move: '+e.message,true);}finally{busy=false;draw();}}
function newGame(){state=makeState();history=[];busy=false;updateRoles();draw();if(!manifest)return;if(!worker){setStatus('Browser workers are unavailable. Host this directory on GitHub Pages.',true);return;}loadSelected();}
function handleCanvas(e){if(busy||!manifest||state.turn!==(humanFirst?0:1)||gameOver())return;const rect=canvas.getBoundingClientRect();const x=(e.clientX-rect.left)/rect.width*FRAME,y=(e.clientY-rect.top)/rect.height*FRAME;const gx=Math.round((x-center)/scale)+7,gy=Math.round((y-center)/scale)+7;if(gx<0||gy<0||gx>=15||gy>=15)return;const a=gy*15+gx;const [px,py]=point(a);if(Math.hypot(x-px,y-py)>scale*.42)return;if(!legal()[a])return;history.push(clone(state));put(a);showTurn();if(!gameOver())setTimeout(aiMove,120);}
function undo(){if(busy||!history.length)return;state=history.pop();newGameStatus();}
function newGameStatus(){draw();showTurn();}
function initializeWorker(){worker=new Worker('./worker.js');worker.onmessage=e=>{const msg=e.data;const job=pending.get(msg.id);if(!job)return;pending.delete(msg.id);if(msg.type==='error')job.reject(new Error(msg.message));else job.resolve(msg);};worker.onerror=e=>{setStatus('Browser inference error: '+e.message,true);};}
el('checkpoint').addEventListener('change',()=>{checkpoint=select.value;newGame();});el('aiFirst').addEventListener('click',()=>{humanFirst=false;newGame();});el('humanFirst').addEventListener('click',()=>{humanFirst=true;newGame();});el('restartBoard').onclick=newGame;el('restartSide').onclick=newGame;el('undoHuman').onclick=undo;canvas.addEventListener('click',handleCanvas);
for(const key of ['showLegal','showGrid','showCoordinates'])el(key).addEventListener('change',draw);
(async()=>{state=makeState();draw();try{const res=await fetch(manifestUrl);if(!res.ok)throw new Error('Could not fetch model manifest');manifest=await res.json();initializeWorker();newGame();}catch(err){el('loading').textContent='Unavailable';setStatus('Could not load the model files. Open this site via GitHub Pages or a local HTTP server, rather than file://. '+err.message,true);}})();
})();
