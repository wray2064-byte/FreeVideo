"""Run separately processable stages of the real experimental VDN-H3 test."""
import argparse, json, os, pathlib, time, threading
import psutil
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['init', 'encode', 'sample', 'decode'])
    parser.add_argument('--prompt', default='A small orange cat turns its head toward the camera in a sunlit garden, gentle natural motion, realistic cinematic video.')
    parser.add_argument('--compute', choices=['vulkan', 'cpu-reference'], default='vulkan')
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    models = root / 'experimental-models'
    output = root / 'experimental-output'
    output.mkdir(exist_ok=True)
    os.environ.setdefault('FREEVIDEO_HOME', str(root / 'experimental-data'))
    os.environ.setdefault('FREEVIDEO_VDN_ROOT', str(root.parent / 'vdn-minimax-h3'))
    os.environ.setdefault('FREEVIDEO_VULKAN_LIBRARY_DIR', str(root.parent / 'runtime'))
    torch.set_num_threads(8)
    started = time.perf_counter()
    if args.stage == 'encode':
        from freevideo_engine.vulkan_stages import encode
        report = encode(args.prompt, output / 'conditioning.pt', root=root / 'experimental-data',
            library=os.environ.get('FREEVIDEO_COMFY_ROOT', str(root.parent / 'ComfyUI')),
            checkpoint=models / 'encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
    elif args.stage in ('init', 'sample'):
        from freevideo_engine.vulkan_runtime import Engine
        engine = Engine(models / 'cache', base=models / 'h3-base', checkpoint=models / 'stage-dmd-step-250',
            steps=8, budget_bytes=4 << 30, query_chunk=256, ff_chunk=1024,
            projection_chunk=1024, attention_chunk=1024, head_chunk=56, compute=args.compute)
        try:
            if args.stage == 'init':
                report = dict(scope='Sampler initialization only', success=True,
                              blocks=len(engine.reference.model.transformer_blocks),
                              load_seconds=engine.reference.load_seconds)
            else:
                video, audio, report = engine.generate(output / 'conditioning.pt', 42,
                    width=256, height=256, frames=39, two_pass=False)
                torch.save(dict(video=video.cpu(), audio=audio.cpu()), output / 'latents.pt')
        finally:
            engine.close()
    else:
        from freevideo_engine.vulkan_stages import decode
        latents = torch.load(output / 'latents.pt', weights_only=True)
        report = decode(latents['video'], latents['audio'], output / f'freevideo-{args.compute}-256.mp4',
            base=models / 'h3-base', artifacts_dir=output / 'decode', compute=args.compute)
    report['stage_wall_seconds'] = time.perf_counter() - started
    report['requested_compute'] = args.compute if args.stage != 'encode' else 'vulkan'
    (output / (args.stage + '-report.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    records, stop = [], threading.Event()
    def monitor():
        process = psutil.Process()
        while not stop.is_set():
            records.append(dict(timestamp=time.time(), process_rss_bytes=process.memory_info().rss,
                                system_available_bytes=psutil.virtual_memory().available))
            stop.wait(2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        main()
    finally:
        stop.set()
        thread.join(timeout=3)
        path = pathlib.Path(__file__).resolve().parents[1] / 'experimental-output'
        path.mkdir(exist_ok=True)
        stage = os.sys.argv[1] if len(os.sys.argv) > 1 else 'unknown'
        (path / (stage + '-memory.json')).write_text(json.dumps(records))
