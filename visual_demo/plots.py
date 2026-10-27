"""Publication plots from archived results, plus live prediction illustrations."""
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from visual_demo.results import GRID, TITLES, RUNS, LABELS
BLUE='#176B87'; ORANGE='#CE642D'; GRAY='#5B6770'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.facecolor':'white'})

def save(fig, stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    for fmt in ['png','pdf']: fig.savefig(str(stem)+'.'+fmt,dpi=300,bbox_inches='tight')

def png_bytes(fig):
    out=io.BytesIO();fig.savefig(out,format='png',dpi=300,bbox_inches='tight');return out.getvalue()

def comparison(data):
    fig,ax=plt.subplots(figsize=(10,5.1));x=np.arange(4);w=.34
    for off,key,label,color in [(-w/2,'clean','Clean inputs',BLUE),(w/2,'mean_degraded_accuracy','Degraded inputs · mean of 18 conditions',ORANGE)]:
        vals=[100*(data[r]['evaluation']['clean']['accuracy'] if key=='clean' else data[r]['evaluation'][key]) for r in RUNS]
        bars=ax.bar(x+off,vals,w,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=9)
    ax.set_xticks(x,LABELS);ax.set_ylim(0,109);ax.set_ylabel('Accuracy (%)');ax.set_title('Degradation-aware training improves robustness of the proposed CNN',pad=16)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False,ncol=2,fontsize=9)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();return fig

def curves(data):
    fig,axs=plt.subplots(2,3,figsize=(12,7.1),sharey=True)
    for ax,(kind,levels) in zip(axs.flat,GRID.items()):
        for run,label,color in [(RUNS[0],LABELS[0],BLUE),(RUNS[1],LABELS[1],ORANGE)]:
            vals=[100*data[run]['evaluation']['per_condition'][f'{kind}/{s}']['accuracy'] for s in levels]
            ax.plot(range(1,4),vals,'o-',color=color,label=label,lw=2)
        ax.set_title(TITLES[kind]);ax.set_xticks([1,2,3],[f'Low\n{s}' if i==0 else f'Medium\n{s}' if i==1 else f'High\n{s}' for i,s in enumerate(levels)])
        ax.set_ylim(0,100);ax.grid(alpha=.15)
    axs[0,0].set_ylabel('Accuracy (%)');axs[1,0].set_ylabel('Accuracy (%)')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Accuracy across degradation severity · archived test results',y=.99)
    fig.tight_layout(rect=[0,.05,1,.96]);return fig

def heatmap(data,run=RUNS[0]):
    vals=np.array([[100*data[run]['evaluation']['per_condition'][f'{kind}/{s}']['accuracy'] for s in levels] for kind,levels in GRID.items()])
    fig,ax=plt.subplots(figsize=(9,5.4));im=ax.imshow(vals,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    for i,(kind,levels) in enumerate(GRID.items()):
        for j,s in enumerate(levels):ax.text(j,i,f'{vals[i,j]:.1f}%\n({s})',ha='center',va='center',fontsize=9,color='white' if vals[i,j]>60 else '#162D3C')
    ax.set_xticks(range(3),['Low severity','Medium severity','High severity']);ax.set_yticks(range(6),list(TITLES.values()))
    ax.set_title(f'{LABELS[RUNS.index(run)]} · 18 degradation conditions',pad=14)
    fig.colorbar(im,ax=ax,label='Accuracy (%)',shrink=.85);fig.tight_layout();return fig

def confusion(data,run=RUNS[0]):
    cm=np.array(data[run]['evaluation']['clean']['confusion_matrix'],dtype=float);den=cm.sum(axis=1,keepdims=True)
    cm=np.divide(cm,den,out=np.zeros_like(cm),where=den!=0)
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
    ax.set_xticks(range(0,43,2));ax.set_yticks(range(0,43,2));ax.set_xlabel('Predicted class ID');ax.set_ylabel('True class ID')
    ax.set_title('Proposed aware · clean-input confusion matrix\nRow-normalized; each row represents one true class')
    fig.colorbar(im,ax=ax,label='Fraction within true class',shrink=.8);fig.tight_layout();return fig

def architecture():
    fig,ax=plt.subplots(figsize=(8.3,10.4));ax.set_xlim(0,10);ax.set_ylim(0,13);ax.axis('off')
    blocks=[('Image preprocessing','RGB → INTER_AREA resize → float32 / 255','64 × 64 × 3'),
            ('Initial convolution','3 × 3, 32 filters, stride 2 · BN · ReLU6','32 × 32 × 32'),
            ('Depthwise separable block 0','3 × 3 depthwise, stride 1 → 1 × 1 pointwise, 32','32 × 32 × 32'),
            ('Depthwise separable block 1','3 × 3 depthwise, stride 2 → 1 × 1 pointwise, 64','16 × 16 × 64'),
            ('Depthwise separable block 2','3 × 3 depthwise, stride 2 → 1 × 1 pointwise, 128','8 × 8 × 128'),
            ('Depthwise separable block 3','3 × 3 depthwise, stride 2 → 1 × 1 pointwise, 256','4 × 4 × 256'),
            ('Classification head','Global average pool → dropout 0.30 → Dense + softmax','43 class probabilities')]
    for i,(a,b,c) in enumerate(blocks):
        y=11.4-i*1.55
        box=FancyBboxPatch((.5,y-.55),9,1.12,boxstyle='round,pad=.12',facecolor='#EFF5F7',edgecolor=BLUE,lw=1.2);ax.add_patch(box)
        ax.text(5,y+.28,a,ha='center',weight='bold',fontsize=12);ax.text(5,y-.06,b,ha='center',fontsize=9);ax.text(5,y-.36,c,ha='center',fontsize=10,color=BLUE)
        if i<6:ax.annotate('',(5,y-1.0),(5,y-.7),arrowprops={'arrowstyle':'->','color':GRAY})
    ax.text(5,12.55,'The supplied lightweight CNN',ha='center',weight='bold',fontsize=17)
    ax.text(5,.65,'Each depthwise and pointwise convolution is followed by BN and ReLU6.\n61,323 total parameters in the default source architecture.\nGrad-CAM target: ds_pw_relu6_3 (4 × 4 spatial resolution).',ha='center',fontsize=10,linespacing=1.7)
    fig.tight_layout();return fig

def live_grid(images, probs, kind, names, truth=None, other_probs=None):
    nr=2 if other_probs is not None else 1
    fig,axs=plt.subplots(nr,4,figsize=(12,3.6*nr),squeeze=False)
    labels=['Clean']+[f'{s} severity' for s in ['Low','Medium','High']]
    for row,pp in enumerate([probs] if other_probs is None else [probs,other_probs]):
        for j in range(4):
            ax=axs[row,j];ax.imshow(images[j],interpolation='nearest');ax.axis('off')
            pred=int(pp[j].argmax());score=float(pp[j,pred]);correct='' if truth is None else (' · correct' if pred==truth else ' · incorrect')
            model_label='Main model' if row==0 else 'Comparison model'
            ax.set_title(f'{model_label} · {labels[j]}\n{names[pred]} · {score:.1%}{correct}',fontsize=10,wrap=True)
    fig.suptitle(f'{TITLES[kind]} · actual checkpoint predictions',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);return fig
