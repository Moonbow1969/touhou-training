import {registerHooks} from 'node:module';
import {existsSync,readFileSync} from 'node:fs';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {resolve} from 'node:path';

const source=resolve(process.argv[2]);
registerHooks({resolve(name,context,next) {
    if(name.endsWith('.js') && name.startsWith('.') && context.parentURL){
        const file=new URL(name.slice(0,-3)+'.ts',context.parentURL);
        if (existsSync(file))return {url:file.href,shortCircuit:true};
    }
    return next(name,context);
}});
const {parseTh07}=await import(pathToFileURL(source).href);
const {additiveKeyDecode,decompress}=await import(new URL('../lzss.ts',pathToFileURL(source)).href);
const rows=[];
for(const path of process.argv.slice(3)){
    const raw=Uint8Array.from(readFileSync(path));
    const meta=parseTh07(raw);
    const data=Uint8Array.from(raw); additiveKeyDecode(data,16,data[13],7);
    const h=new DataView(data.buffer);
    const vec=decompress(data.slice(84),h.getUint32(20,true),h.getUint32(24,true));
    const body=new DataView(vec.buffer,vec.byteOffset,vec.byteLength);
    const offs=Array.from({length:14},(_,i)=>h.getUint32(28+i*4,true));
    const stages=[];
    for(let i=0;i<2;i++){
        const start=offs[i];
        if(!start)continue;
        const end=Math.min(...[...offs,vec.length+84].filter(x=>x>start));
        const first=start-84+44;
        let fire=0,move=0,frames=0;
        for(let pos=first;pos+4<=end-84;pos+=4){
            const word=body.getUint16(pos,true);fire+=Boolean(word&1);move+=Boolean(word&240);frames++;
        }
        stages.push({stage:i+1,complete_input_records:frames,fire_records:fire,direction_records:move});
    }
    rows.push({path:path.split(/[\\/]/).at(-1),player:meta.player,shot:meta.character,difficulty:meta.difficulty,stages_in_independent_parser:meta.splits.map(s=>s.stage),stage_input_evidence:stages,major:body.getUint16(22,true),minor:vec[1],embedded_version:Buffer.from(vec.slice(140,145)).toString('ascii'),exe_bytes:body.getUint32(132,true),exe_checksum:body.getUint32(136,true).toString(16).padStart(8,'0'),archive_score:meta.score});
}
console.log(JSON.stringify(rows,null,2));
