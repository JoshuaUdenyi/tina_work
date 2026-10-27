"""Export archived thesis figures, and optionally genuine checkpoint illustrations.

python export_figures.py
python export_figures.py --image sign.png --checkpoint checkpoints/proposed_aware/best.keras --true-class 14
"""
from pathlib import Path
import argparse,json,hashlib
import matplotlib.pyplot as plt
from visual_demo.results import ROOT,GRID,load_results,audit_results
from visual_demo import plots

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'thesis_figures'))
    ap.add_argument('--image');ap.add_argument('--checkpoint');ap.add_argument('--true-class',type=int,choices=range(43))
    ap.add_argument('--weights-only',action='store_true');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--class-names');args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=load_results();checks=audit_results(data)
    builders={'01_architecture':plots.architecture,'02_model_comparison':lambda:plots.comparison(data),
              '03_robustness_curves':lambda:plots.curves(data),'04_condition_heatmap':lambda:plots.heatmap(data),
              '05_confusion_clean':lambda:plots.confusion(data)}
    for name,build in builders.items():
        fig=build();plots.save(fig,out/name);plt.close(fig);print('Saved',name)
    manifest={'source':'Archived evaluation JSON files from the user supplied Tina (3).zip',
              'checks':checks,'files':{},'new_inference':False}
    for run in data:
        path=ROOT/'results'/run/'eval.json';manifest['files'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if bool(args.image)!=bool(args.checkpoint):ap.error('--image and --checkpoint must be supplied together')
    if args.image:
        from visual_demo.core import preprocess_bytes,load_checkpoint,predict,variants,class_names,fingerprint,gradcam
        model=load_checkpoint(args.checkpoint,'Proposed weights only' if args.weights_only else 'Full saved model')
        raw=Path(args.image).read_bytes();image=preprocess_bytes(raw);names=class_names(args.class_names)
        outputs={}
        for kind in GRID:
            imgs=variants(image,kind,args.seed);prob=predict(model,imgs)
            fig=plots.live_grid(imgs,prob,kind,names,args.true_class);plots.save(fig,out/f'live_{kind}');plt.close(fig)
            outputs[kind]={'probabilities':prob.tolist(),'predictions':prob.argmax(axis=1).tolist()}
        manifest.update(new_inference=True,checkpoint_sha256=fingerprint(args.checkpoint),
                        input_sha256=hashlib.sha256(raw).hexdigest(),seed=args.seed,true_class=args.true_class,
                        predictions=outputs)
        if model.name=='proposed_lightweight_cnn':
            target=args.true_class if args.true_class is not None else int(predict(model,image[None])[0].argmax())
            hm=gradcam(model,image,target);fig,ax=plt.subplots(figsize=(5,5));ax.imshow(image);ax.imshow(hm,extent=(-.5,63.5,63.5,-.5),cmap='inferno',alpha=.45)
            ax.set_title(f'Grad-CAM · {names[target]} · 4 × 4 activation');ax.axis('off');plots.save(fig,out/'live_gradcam_clean');plt.close(fig)
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Figures saved to {out.resolve()}')
if __name__=='__main__':main()
