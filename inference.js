/* Pure JavaScript inference for the exact PyTorch PolicyValueNet architecture.
 * All tensors use channel-first contiguous Float32Array layout. */
(function (root) {
  'use strict';
  const N = 15, PIX = N*N, WIDTH = 64;
  function unpack(buffer, meta) {
    const arrays = {};
    const src = new Float32Array(buffer);
    for (const [key, range] of Object.entries(meta)) arrays[key] = src.subarray(range[0], range[0] + range[1]);
    return arrays;
  }
  // 3x3 zero-padded convolution, NCHW, batch size 1.
  function conv3(input, W, B, ins, outs) {
    const output = new Float32Array(outs * PIX);
    for (let o=0; o<outs; o++) {
      const ob=o*PIX;
      output.fill(B[o],ob,ob+PIX);
      for (let i=0; i<ins; i++) {
        const ib=i*PIX, wb=(o*ins+i)*9;
        for (let y=0; y<N; y++) {
          const row=y*N;
          for (let x=0; x<N; x++) {
            const val=input[ib+row+x];
            if (val===0) continue;
            for (let ky=0; ky<3; ky++) {
              const yy=y+ky-1;
              if (yy<0||yy>=N) continue;
              for (let kx=0; kx<3; kx++) {
                const xx=x+kx-1;
                if (xx<0||xx>=N) continue;
                output[ob+yy*N+xx] += val * W[wb+(2-ky)*3+(2-kx)];
              }
            }
          }
        }
      }
    }
    return output;
  }
  function reluInPlace(a, residual=null) {
    if (residual) for(let i=0;i<a.length;i++) a[i]=Math.max(0,a[i]+residual[i]);
    else for(let i=0;i<a.length;i++) if(a[i]<0)a[i]=0;
    return a;
  }
  function conv1(input,W,B,ins,outs) {
    const output=new Float32Array(outs*PIX);
    for(let o=0;o<outs;o++) for(let p=0;p<PIX;p++) {
      let s=B[o]; for(let i=0;i<ins;i++)s+=W[o*ins+i]*input[i*PIX+p];
      output[o*PIX+p]=s;
    }
    return output;
  }
  function linear(input,W,B,outs) {
    const len=input.length, output=new Float32Array(outs);
    for(let o=0;o<outs;o++) {
      let v=B[o],off=o*len;
      for(let i=0;i<len;i++)v+=W[off+i]*input[i];
      output[o]=v;
    }
    return output;
  }
  function logits(w, obs) {
    let h=reluInPlace(conv3(obs,w['start.weight'],w['start.bias'],4,WIDTH));
    for(let b=0;b<4;b++) {
      let a=reluInPlace(conv3(h,w[`trunk.${b}.a.weight`],w[`trunk.${b}.a.bias`],WIDTH,WIDTH));
      let next=conv3(a,w[`trunk.${b}.b.weight`],w[`trunk.${b}.b.bias`],WIDTH,WIDTH);
      h=reluInPlace(next,h);
    }
    const mid=conv1(h,w['policy.0.weight'],w['policy.0.bias'],WIDTH,2);
    return linear(mid,w['policy.2.weight'],w['policy.2.bias'],PIX);
  }
  function legal(board,coords,occupants,radius) {
    const allowed=board.slice();
    const limit=(2*radius)**2-1e-8;
    for(const a of occupants) {
      const [ax,ay]=coords[a];
      for(let i=0;i<PIX;i++) {
        const [x,y]=coords[i];
        if((x-ax)**2+(y-ay)**2<limit)allowed[i]=0;
      }
    }
    return allowed;
  }
  function geometry(radius=1.7,outer=7,hole=0) {
    const coords=Array.from({length:PIX},(_,i)=>[i%N-7,Math.floor(i/N)-7]);
    const board=Uint8Array.from(coords,([x,y])=>{
      const d=Math.hypot(x,y);
      return +(d<=outer-radius+1e-8 && (!hole||d>=hole+radius-1e-8));
    });
    return {coords,board,radius,outer,hole};
  }
  function observation(state,geom) {
    const o=new Float32Array(4*PIX);
    const own=state.coins[state.turn], other=state.coins[1-state.turn];
    for(const a of own)o[a]=1;
    for(const a of other)o[PIX+a]=1;
    if(state.last>=0)o[PIX*2+state.last]=1;
    o.set(geom.board,PIX*3);
    return o;
  }
  function choose(scores,allowed) {
    let chosen=-1,top=-Infinity;
    for(let i=0;i<PIX;i++) if(allowed[i]&&scores[i]>top) {top=scores[i];chosen=i;}
    return chosen;
  }
  root.CoinInference={unpack,logits,legal,geometry,observation,choose,N,PIX};
})(typeof self !== 'undefined' ? self : globalThis);
