const FP16_BYTES = 2;

function conv(name, cin, cout, h, w, k = 3, stride = 1, pool = 1) {
  const oh = Math.floor((h - k + 2 * Math.floor(k / 2)) / stride + 1);
  const ow = Math.floor((w - k + 2 * Math.floor(k / 2)) / stride + 1);
  return { name, type:"conv", fwd:2*oh*ow*cout*cin*k*k, bwd:4*oh*ow*cout*cin*k*k, params:cin*cout*k*k+cout, out:Math.ceil(oh/pool)*Math.ceil(ow/pool)*cout };
}
function fc(name, input, output) { return { name, type:"fc", fwd:2*input*output, bwd:4*input*output, params:input*output+output, out:output }; }

const architectures = {
  small: {
    label:"Small CNN", hint:"6 compute blocks · image input 32 × 32 × 3",
    layers:[conv("conv1",3,32,32,32),conv("conv2",32,32,32,32,3,1,2),conv("conv3",32,64,16,16),conv("conv4",64,64,16,16,3,1,2),fc("fc1",4096,512),fc("fc2",512,10)]
  },
  alexnet: {
    label:"AlexNet-style CNN", hint:"8 compute blocks · image input 227 × 227 × 3",
    layers:[
      {...conv("conv1",3,96,227,227,11,4),out:27*27*96},
      {...conv("conv2",96,256,27,27,5),out:13*13*256},
      conv("conv3",256,384,13,13),conv("conv4",384,384,13,13),
      {...conv("conv5",384,256,13,13),out:6*6*256},
      fc("fc6",9216,4096),fc("fc7",4096,4096),fc("fc8",4096,1000)
    ]
  },
  vgg: {
    label:"VGG-style CNN", hint:"16 compute blocks · image input 224 × 224 × 3",
    layers:[
      conv("c1·1",3,64,224,224),conv("c1·2",64,64,224,224,3,1,2),
      conv("c2·1",64,128,112,112),conv("c2·2",128,128,112,112,3,1,2),
      conv("c3·1",128,256,56,56),conv("c3·2",256,256,56,56),conv("c3·3",256,256,56,56,3,1,2),
      conv("c4·1",256,512,28,28),conv("c4·2",512,512,28,28),conv("c4·3",512,512,28,28,3,1,2),
      conv("c5·1",512,512,14,14),conv("c5·2",512,512,14,14),conv("c5·3",512,512,14,14,3,1,2),
      fc("fc6",25088,4096),fc("fc7",4096,4096),fc("fc8",4096,1000)
    ]
  },
  ffn: {
    label:"Deep feed-forward network", hint:"12 fully connected blocks · width 4096",
    layers:Array.from({length:12},(_,i)=>fc(i===11?"output":`fc${i+1}`,4096,i===11?1000:4096))
  }
};

const controls = {
  architecture:document.querySelector("#architecture"), gpus:document.querySelector("#gpu-count"), tflops:document.querySelector("#tflops"), bandwidth:document.querySelector("#bandwidth"), batch:32
};
let lastResult = null;

function fmtTime(ms) {
  if (ms < .001) return `${(ms*1e6).toFixed(1)} ns`;
  if (ms < 1) return `${(ms*1000).toFixed(ms < .1 ? 1 : 0)} µs`;
  return `${ms.toFixed(ms < 10 ? 2 : 1)} ms`;
}

function buildCosts(layers, batch, tflops, bandwidth) {
  const flopRate = tflops * 1e12;
  const byteRate = bandwidth * 1e9;
  return {
    compute:layers.map(l => batch*(l.fwd+l.bwd)/flopRate*1000),
    boundary:layers.map(l => batch*l.out*FP16_BYTES/byteRate*1000),
    params:layers.map(l => l.params*FP16_BYTES)
  };
}

function optimize(layers, gpuCount, batch, tflops, bandwidth) {
  const n=layers.length, costs=buildCosts(layers,batch,tflops,bandwidth), inf=Number.POSITIVE_INFINITY;
  const dp=Array.from({length:n+1},()=>Array(gpuCount+1).fill(inf));
  const choice=Array.from({length:n+1},()=>Array(gpuCount+1).fill(null));
  const prefixCompute=[0], prefixParams=[0];
  costs.compute.forEach((v,i)=>{prefixCompute.push(prefixCompute[i]+v);prefixParams.push(prefixParams[i]+costs.params[i]);});
  const stageCost=(start,end,r)=>{
    const compute=(prefixCompute[end+1]-prefixCompute[start])/r;
    const paramBytes=prefixParams[end+1]-prefixParams[start];
    const sync=r===1?0:(2*(r-1)/r*paramBytes)/(bandwidth*1e9)*1000;
    return { total:Math.max(compute,sync), compute, sync };
  };
  for(let j=1;j<=n;j++){
    for(let m=1;m<=gpuCount;m++){
      const whole=stageCost(0,j-1,m); dp[j][m]=whole.total; choice[j][m]={type:"single",start:0,end:j-1,replicas:m,stage:whole};
      for(let i=1;i<j;i++){
        for(let r=1;r<m;r++){
          if(!Number.isFinite(dp[i][m-r])) continue;
          const tail=stageCost(i,j-1,r), comm=2*costs.boundary[i-1];
          const candidate=Math.max(dp[i][m-r],comm,tail.total);
          if(candidate<dp[j][m]){dp[j][m]=candidate;choice[j][m]={type:"split",cut:i,leftGPUs:m-r,replicas:r,comm,stage:tail};}
        }
      }
    }
  }
  const recover=(j,m)=>{const c=choice[j][m];if(c.type==="single")return[{start:0,end:j-1,replicas:m,...c.stage,boundary:0}];return[...recover(c.cut,c.leftGPUs),{start:c.cut,end:j-1,replicas:c.replicas,...c.stage,boundary:c.comm}];};
  const dataParallelTime=stageCost(0,n-1,gpuCount).total;
  return {dp,choice,stages:recover(n,gpuCount),costs,bottleneck:dp[n][gpuCount],dataParallelTime,layers,gpuCount,batch};
}

function renderPlan(result) {
  const {layers,stages,bottleneck,batch}=result;
  document.querySelector("#plan-title").textContent=`${architectures[controls.architecture.value].label} on ${result.gpuCount} GPU${result.gpuCount>1?"s":""}`;
  document.querySelector("#bottleneck-value").textContent=fmtTime(bottleneck);
  document.querySelector("#throughput-value").textContent=Math.round(batch/(bottleneck/1000)).toLocaleString();
  document.querySelector("#stage-count-value").textContent=stages.length;
  document.querySelector("#speedup-value").textContent=`${(result.dataParallelTime/bottleneck).toFixed(2)}×`;
  document.querySelector("#speedup-baseline").textContent=`DP: ${Math.round(batch*1000/result.dataParallelTime).toLocaleString()} samples/s`;
  const replicated=stages.filter(s=>s.replicas>1).length;
  document.querySelector("#replica-summary").textContent=replicated?`${replicated} replicated stage${replicated>1?"s":""}`:"no stage replication";

  renderGpuGraph(result);

}

function renderGpuGraph(result) {
  const {stages,layers,bottleneck}=result,svg=document.querySelector("#gpu-graph"),ns="http://www.w3.org/2000/svg";
  const previousPositions=new Map(Array.from(svg.querySelectorAll(".gpu-node")).map(node=>[
    Number(node.dataset.gpu), {x:Number(node.dataset.x),y:Number(node.dataset.y)}
  ]));
  const widths=stages.map(s=>Math.max(145,Math.ceil(Math.sqrt(s.replicas))*70+34));
  const gap=64,pad=20,naturalWidth=widths.reduce((a,b)=>a+b,0)+gap*(stages.length-1)+pad*2;
  const available=svg.parentElement.clientWidth,vertical=naturalWidth>available;
  const totalWidth=vertical?Math.max(...widths)+40:naturalWidth,height=vertical?stages.length*350:330;
  svg.setAttribute("viewBox",`0 0 ${totalWidth} ${height}`);svg.style.minWidth="0";svg.style.height=`${vertical?height*Math.min(1,available/totalWidth):Math.max(330,svg.parentElement.clientHeight)}px`;svg.innerHTML=`<defs><marker id="graph-fwd" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7z" fill="#47d7ac"/></marker><marker id="graph-back" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7z" fill="#ff705e"/></marker></defs>`;
  const layouts=[];let cursor=pad,gpuId=0;
  stages.forEach((stage,stageIndex)=>{
    if(vertical)cursor=(totalWidth-widths[stageIndex])/2;
    const width=widths[stageIndex],cols=Math.ceil(Math.sqrt(stage.replicas)),rows=Math.ceil(stage.replicas/cols),nodeW=54,nodeH=42,xGap=12,yGap=13;
    const gridW=cols*nodeW+(cols-1)*xGap,gridH=rows*nodeH+(rows-1)*yGap,startX=cursor+(width-gridW)/2,startY=72+(205-gridH)/2;
    const nodes=[];for(let r=0;r<stage.replicas;r++){const col=r%cols,row=Math.floor(r/cols);nodes.push({x:startX+col*(nodeW+xGap),y:startY+row*(nodeH+yGap),gpu:gpuId++,replica:r});}
    layouts.push({x:cursor,width,nodes,center:cursor+width/2,stage,stageIndex});cursor+=width+gap;
  });
  const bottleneckIndex=stages.findIndex(s=>Math.abs(Math.max(s.compute,s.sync,s.boundary)-bottleneck)<1e-9);
  layouts.slice(0,-1).forEach((layout,i)=>{const next=layouts[i+1],x1=layout.x+layout.width,x2=next.x,y=165;
    if(vertical){const x=totalWidth/2,top=i*350+307,bottom=(i+1)*350+22;svg.insertAdjacentHTML("beforeend",`<path class="pipeline-edge forward" d="M${x-8} ${top} V${bottom}" marker-end="url(#graph-fwd)"/><path class="pipeline-edge backward" d="M${x+8} ${bottom} V${top}" marker-end="url(#graph-back)"/><text class="edge-label forward" x="${x-18}" y="${top+30}" text-anchor="end">ACTIVATIONS ↓</text><text class="edge-label backward" x="${x+18}" y="${top+30}">↑ GRADIENTS</text>`);return;}
    svg.insertAdjacentHTML("beforeend",`<path class="pipeline-edge forward" d="M${x1} ${y-6} H${x2}" marker-end="url(#graph-fwd)"/><path class="pipeline-edge backward" d="M${x2} ${y+7} H${x1}" marker-end="url(#graph-back)"/>`);});
  layouts.forEach(layout=>{
    const before=new Set(svg.children);
    const {stage,stageIndex,nodes}=layout,layerLabel=stage.start===stage.end?layers[stage.start].name:`${layers[stage.start].name} → ${layers[stage.end].name}`;
    svg.insertAdjacentHTML("beforeend",`<rect class="stage-cluster${stageIndex===bottleneckIndex?" bottleneck":""}" x="${layout.x}" y="22" width="${layout.width}" height="285"/><text class="stage-title" x="${layout.x+12}" y="42">STAGE ${stageIndex+1} · ${stage.replicas} GPU${stage.replicas>1?"s":""}</text><text class="stage-subtitle" x="${layout.x+12}" y="56">${layerLabel}</text>`);
    nodes.forEach((n,r)=>{
      const effective=Math.max(stage.compute,stage.sync,stage.boundary),isBottleneck=Math.abs(effective-bottleneck)<1e-9;
      const group=document.createElementNS(ns,"g"),old=previousPositions.get(n.gpu);
      group.setAttribute("class",`gpu-node${isBottleneck?" bottleneck":""}${old?"":" entering"}`);
      group.dataset.gpu=n.gpu;group.dataset.x=n.x;group.dataset.y=n.y;
      group.setAttribute("tabindex","0");group.setAttribute("role","button");group.setAttribute("aria-label",`GPU ${n.gpu}, stage ${stageIndex+1}, replica ${r+1}`);
      if(old)group.style.transform=`translate(${old.x-n.x}px,${old.y-n.y}px)`;else group.style.animationDelay=`${n.gpu*25}ms`;
      group.innerHTML=`<rect x="${n.x}" y="${n.y}" width="54" height="42"/><rect class="gpu-chip" x="${n.x+7}" y="${n.y+8}" width="4" height="18"/><text class="gpu-id" x="${n.x+17}" y="${n.y+18}">GPU ${n.gpu}</text><text class="gpu-stage" x="${n.x+17}" y="${n.y+31}">S${stageIndex+1} · R${r+1}</text>`;
      const select=()=>selectGpu(result,stageIndex,r,n.gpu,group);group.addEventListener("click",select);group.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select();}});svg.appendChild(group);
      if(old)requestAnimationFrame(()=>requestAnimationFrame(()=>{group.style.transform="translate(0,0)";}));
    });
    if(vertical){const wrapper=document.createElementNS(ns,"g");wrapper.setAttribute("transform",`translate(0 ${stageIndex*350})`);Array.from(svg.children).filter(el=>!before.has(el)).forEach(el=>wrapper.appendChild(el));svg.appendChild(wrapper);}
  });
  const defaultLayout=layouts[bottleneckIndex]||layouts[0],defaultNode=defaultLayout.nodes[0],defaultEl=svg.querySelector(`[aria-label^="GPU ${defaultNode.gpu},"]`);selectGpu(result,defaultLayout.stageIndex,0,defaultNode.gpu,defaultEl);
}

function selectGpu(result,stageIndex,replica,gpuId,element){
  document.querySelectorAll(".gpu-node.selected").forEach(n=>n.classList.remove("selected"));if(element)element.classList.add("selected");
  const stage=result.stages[stageIndex],names=result.layers.slice(stage.start,stage.end+1).map(l=>l.name),effective=Math.max(stage.compute,stage.sync,stage.boundary);
  document.querySelector("#selected-gpu").textContent=`GPU ${gpuId}`;document.querySelector("#selected-assignment").textContent=`Stage ${stageIndex+1} · replica ${replica+1} of ${stage.replicas} · stage time ${fmtTime(effective)}`;
  document.querySelector("#selected-layers").textContent=names.length>4?`${names[0]} → ${names[names.length-1]} (${names.length} blocks)`:names.join(", ");
  document.querySelector("#selected-compute").textContent=fmtTime(stage.compute);document.querySelector("#selected-sync").textContent=stage.replicas>1?fmtTime(stage.sync):"none";document.querySelector("#selected-boundary").textContent=stageIndex?fmtTime(stage.boundary):"input stage";
}

function run(animate=false,includeMatrix=true) {
  const arch=architectures[controls.architecture.value],gpus=Number(controls.gpus.value),tflops=Number(controls.tflops.value),bandwidth=Number(controls.bandwidth.value);
  lastResult=optimize(arch.layers,gpus,controls.batch,tflops,bandwidth);renderPlan(lastResult);
}

function syncLabels(){document.querySelector("#gpu-count-value").textContent=controls.gpus.value;document.querySelector("#tflops-value").textContent=controls.tflops.value;document.querySelector("#bandwidth-value").textContent=controls.bandwidth.value;document.querySelector("#microbatch-value").textContent=controls.batch;document.querySelector("#architecture-hint").textContent=architectures[controls.architecture.value].hint;}
let liveRenderFrame=null;
function scheduleLiveRender(){
  syncLabels();cancelAnimationFrame(liveRenderFrame);
  liveRenderFrame=requestAnimationFrame(()=>run(false,false));
}
[controls.gpus,controls.tflops,controls.bandwidth].forEach(input=>{
  input.addEventListener("input",scheduleLiveRender);
  input.addEventListener("change",()=>{cancelAnimationFrame(liveRenderFrame);run(false);});
});
controls.architecture.addEventListener("change",()=>{syncLabels();run(true);});
document.querySelectorAll("[data-batch]").forEach(btn=>btn.addEventListener("click",()=>{controls.batch=Number(btn.dataset.batch);document.querySelectorAll("[data-batch]").forEach(b=>b.classList.toggle("active",b===btn));syncLabels();run(false);}));
document.querySelector("#optimize-btn").addEventListener("click",()=>{
  const btn=document.querySelector("#optimize-btn"),status=document.querySelector(".status-pill"),shell=document.querySelector(".graph-shell"),label=btn.querySelector("span");
  btn.classList.add("running");status.classList.add("running");shell.classList.add("solving");label.textContent="SOLVING…";
  document.querySelector("#status-copy").textContent="EVALUATING SUBPROBLEMS";
  document.querySelector(".gpu-topology").scrollIntoView({behavior:"smooth",block:"center"});
  setTimeout(()=>{
    document.querySelector("#gpu-graph").innerHTML="";run(true);
    btn.classList.remove("running");status.classList.remove("running");shell.classList.remove("solving");label.textContent="RUN DYNAMIC PROGRAM";
    document.querySelector("#status-copy").textContent="OPTIMAL PLAN READY";
  },650);
});

syncLabels();run(true);
let resizeFrame;
window.addEventListener("resize",()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(lastResult)renderGpuGraph(lastResult);});});
