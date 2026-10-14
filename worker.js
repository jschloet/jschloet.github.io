importScripts('./inference.js');
const I=self.CoinInference;
const cache=new Map();
self.onmessage=async ({data})=>{
  const {id,type}=data;
  try {
    if(type==='load') {
      const key=data.checkpoint;
      if(!cache.has(key)) {
        const response=await fetch(`./models/${key}.bin`);
        if(!response.ok)throw new Error(`Could not load model (${response.status})`);
        cache.set(key,I.unpack(await response.arrayBuffer(),data.meta));
      }
      self.postMessage({id,type:'loaded',checkpoint:key});
    } else if(type==='predict') {
      const model=cache.get(data.checkpoint);
      if(!model)throw new Error('Checkpoint not loaded');
      const geom=I.geometry(data.radius,data.outer,data.hole);
      const obs=I.observation(data.state,geom);
      const scores=I.logits(model,obs);
      const allowed=I.legal(geom.board,geom.coords,[...data.state.coins[0],...data.state.coins[1]],data.radius);
      const action=I.choose(scores,allowed);
      self.postMessage({id,type:'predicted',action,scores:Array.from(scores)});
    }
  } catch(e){self.postMessage({id,type:'error',message:String(e.message||e)});}
};
