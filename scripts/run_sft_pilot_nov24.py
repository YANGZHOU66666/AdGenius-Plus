"""Paired local Qwen3.5 SFT pilot. Run with PYTHONPATH=src and mlx-lm 0.32.2."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.tuner.utils import linear_to_lora_layers
from adgenius.shared.product import render_product_information
from generate_candidates import SYSTEM_PROMPT

SEED = 20261124
TRAIN_N, DEV_N = 128, 32
MAX_NEW = 96
LORA = {"rank":8, "scale":8.0, "dropout":0.0}


def read(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def save_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False),file=f)
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def examples(data, split, n, tok):
    teachers = {r["product_id"]:r["generation_result"] for r in read(data/f"teacher_qwen397b_{split}.jsonl")}
    result = []
    for p in read(data/f"{split}_products.jsonl")[:n]:
        teacher = teachers[p["product_id"]]
        assert teacher["status"] == "success"
        ref = teacher["text"]
        msgs = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":render_product_information(p)}]
        prompt = tok.apply_chat_template(msgs,add_generation_prompt=True,enable_thinking=False)
        target = tok.encode(ref,add_special_tokens=False)+[tok.eos_token_id]
        assert len(prompt)+len(target)<8192 and ref.strip()
        result.append((p,prompt,target,ref))
    assert len(result)==n
    return result


def loss(model, prompt, target):
    # Causal LM teacher forcing. Project only states predicting a response token.
    ids = mx.array([prompt+target[:-1]])
    textmodel = model.language_model
    hid = textmodel.model(ids)[:,len(prompt)-1:,:]
    logits = textmodel.model.embed_tokens.as_linear(hid) if textmodel.args.tie_word_embeddings else textmodel.lm_head(hid)
    return nn.losses.cross_entropy(logits.astype(mx.float32),mx.array([target]),reduction="mean")


def evaluate(model, tok, dev, name, out):
    model.eval()
    rows=[]
    for i in range(0,len(dev),4):
        part=dev[i:i+4]
        resp=batch_generate(model,tok,[x[1] for x in part],max_tokens=MAX_NEW,return_token_ids=True,verbose=False)
        for (p,prompt,target,ref),ad,ids in zip(part,resp.texts,resp.token_ids,strict=True):
            nll=float(loss(model,prompt,target).item())
            assert math.isfinite(nll)
            cid="cand_"+hashlib.sha256((name+p["product_id"]).encode()).hexdigest()[:24]
            rows.append(dict(candidate_id=cid,product_id=p["product_id"],run_id=name,sample_index=0,
                             generation_result=dict(status="success" if ad.strip() else "failed",text=ad if ad.strip() else None,error=None if ad.strip() else "empty_output"),
                             trace_id=None))
            stats.append(dict(system=name,product_id=p["product_id"],nll=nll,target_tokens=len(target),output_tokens=len(ids),words=len(ad.split()),at_limit=len(ids)>=MAX_NEW))
        print(name, i+len(part),"/",len(dev),flush=True)
    save_rows(out/f"results_{name}.jsonl",rows)
    return rows


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("results/sft_pilot_nov24"))
    ap.add_argument("--adapters",type=Path,default=Path("artifacts/pilot_nov24/adapter"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    args.adapters.mkdir(parents=True,exist_ok=True)
    if (args.out/"stats.jsonl").exists():
        raise SystemExit("Completed results exist. Use a fresh output directory to rerun.")
    mx.random.seed(SEED)
    if digest(args.model/"model.safetensors") != "f5a0d9dd3efa73510542a8023d610ff26be2b4b020d181cfc4bedaa1fcc5dd9e":
        raise ValueError("Unexpected snapshot weights")
    model,tok=load(str(args.model))
    model.eval()
    train=examples(args.data,"train",TRAIN_N,tok)
    dev=examples(args.data,"dev",DEV_N,tok)
    selected=set(p["source_id"] for p,*_ in train)
    assert len(selected)==TRAIN_N and selected.isdisjoint(p["source_id"] for p,*_ in dev)
    save_rows(args.out/"dev_products.jsonl",[v[0] for v in dev])
    stats=[]
    before=evaluate(model,tok,dev,"base",args.out)
    save_rows(args.out/"base_stats.jsonl",stats)
    print("Starting SFT",flush=True)
    model.freeze()
    assert not model.layers[-1].is_linear
    # Last full-attention block only. Other blocks stay frozen in inference mode.
    linear_to_lora_layers(model,1,LORA)
    model.eval()
    trainable=sum(p.size for _,p in tree_flatten(model.trainable_parameters()))
    assert 0<trainable<3000000
    opt=optim.AdamW(learning_rate=1e-4,weight_decay=0.01)
    grad=nn.value_and_grad(model,loss)
    random.Random(SEED).shuffle(train)
    log=[]
    start=time.monotonic()
    for step,(p,prompt,target,_) in enumerate(train,1):
        val,grads=grad(model,prompt,target)
        opt.update(model,grads)
        mx.eval(model.trainable_parameters(),opt.state,val)
        value=float(val.item())
        assert math.isfinite(value)
        log.append(dict(step=step,product_id=p["product_id"],nll=value,target_tokens=len(target)))
        if step%8==0:print("SFT step",step,"loss",round(value,3),"seconds",int(time.monotonic()-start),flush=True)
    elapsed=time.monotonic()-start
    save_rows(args.out/"training_log.jsonl",log)
    (args.adapters/"adapter_config.json").write_text(json.dumps(dict(fine_tune_type="lora",num_layers=1,lora_parameters=LORA),indent=2))
    mx.save_safetensors(str(args.adapters/"adapters.safetensors"),dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(args.adapters/"adapters.safetensors"),dict(tree_flatten(model.trainable_parameters())))
    (args.adapters/"adapter_config.json").write_text(json.dumps(dict(fine_tune_type="lora",num_layers=1,lora_parameters=LORA)))
    after=evaluate(model,tok,dev,"sft",args.out)
    save_rows(args.out/"stats.jsonl",stats)
    rng=random.Random(SEED)
    pairs=[]
    for a,b in zip(before,after,strict=True):
        assert a["product_id"]==b["product_id"]
        if any(r["generation_result"]["status"]!="success" for r in [a,b]):continue
        cs=[dict(candidate_id=r["candidate_id"],text=r["generation_result"]["text"]) for r in [a,b]]
        rng.shuffle(cs)
        pairs.append(dict(pair_id="pair_"+a["product_id"],product_id=a["product_id"],candidates=cs))
    save_rows(args.out/"candidate_pairs.jsonl",pairs)
    manifest=dict(seed=SEED,train=TRAIN_N,dev=DEV_N,trainable_params=trainable,train_seconds=elapsed,
                  model="mlx-community/Qwen3.5-0.8B-4bit",revision="da28692b5f139cb0ec58a356b437486b7dac7462",
                  decoding="greedy",max_new_tokens=MAX_NEW,lr=1e-4,lora=LORA,batch_size=1,
                  adapter_sha256=digest(args.adapters/"adapters.safetensors"),
                  source_hashes={f.name:digest(f) for f in args.data.glob("*.jsonl")},
                  base_sha256=digest(args.model/"model.safetensors"))
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Done",flush=True)
