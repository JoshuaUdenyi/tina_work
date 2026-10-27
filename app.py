# """Launch: python -m streamlit run app.py"""
# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# from visual_demo.results import ROOT, GRID, TITLES, RUNS, LABELS, load_results, audit_results, rows
# from visual_demo import plots
# st.set_page_config(page_title='TINA | Traffic sign robustness',layout='wide')
# st.title('Traffic sign recognition under degraded conditions')
# st.caption('TINA thesis demonstration · TensorFlow/Keras · 64 × 64 RGB · 43 classes')
# results_tab, live_tab, architecture_tab = st.tabs(['Saved experiment results','Live image demonstration','How the model works'])
# data=load_results()
# with results_tab:
#     st.info('These are archived metrics from your uploaded ZIP, not new inference results. No checkpoint is needed for this tab.')
#     try: audit_notes=audit_results(data)
#     except AssertionError as exc: st.error(str(exc));st.stop()
#     main=data[RUNS[0]]['evaluation'];base=data[RUNS[1]]['evaluation']
#     c=st.columns(4)
#     c[0].metric('Clean accuracy',f"{main['clean']['accuracy']:.2%}")
#     c[1].metric('Mean degraded accuracy',f"{main['mean_degraded_accuracy']:.2%}",f"{100*(main['mean_degraded_accuracy']-base['mean_degraded_accuracy']):.2f} percentage points vs clean-only")
#     c[2].metric('Robustness R',f"{main['robustness_score']:.3f}")
#     c[3].metric('Archived test samples',f"{data[RUNS[0]]['num_test']:,}")
#     st.caption('R = mean accuracy across 18 degraded conditions ÷ clean accuracy. Softmax confidence on one image is a different quantity.')
#     st.dataframe(pd.DataFrame(rows(data)),hide_index=True)
#     kind=st.selectbox('Inspect a degradation',list(GRID),format_func=lambda k:TITLES[k])
#     chart={LABELS[i]:[100*data[r]['evaluation']['per_condition'][f'{kind}/{s}']['accuracy'] for s in GRID[kind]] for i,r in enumerate(RUNS)}
#     st.line_chart(pd.DataFrame(chart,index=['Low','Medium','High']),y_label='Accuracy (%)',x_label='Severity')
#     st.caption(f'Actual settings: {GRID[kind]}')
#     fig=plots.comparison(data);st.pyplot(fig);plt.close(fig)
#     with st.expander('Result consistency and provenance'):
#         st.write('\n\n'.join(audit_notes))
#         st.write('Checks confirm arithmetic consistency only. The original dataset, split manifest and trained checkpoints were not included, so archived experimental provenance cannot be independently reproduced here. The dummy smoke-test run is excluded.')
#     for name in ['01_architecture','02_model_comparison','03_robustness_curves','04_condition_heatmap','05_confusion_clean']:
#         f=ROOT/'thesis_figures'/f'{name}.png'
#         if f.exists():st.download_button(f'Download {name}',f.read_bytes(),file_name=f.name,mime='image/png')

# @st.cache_resource
# def cached_model(path,mode,size,mtime):
#     from visual_demo.core import load_checkpoint
#     return load_checkpoint(path,mode)

# with live_tab:
#     st.write('Upload a tightly cropped traffic-sign image. Adjust the degradation and examine the model response. This classifier does not locate signs in full road scenes.')
#     path=st.text_input('Main checkpoint path',str(ROOT/'checkpoints/proposed_aware/best.keras'))
#     mode=st.selectbox('Checkpoint format',['Full saved model','Proposed weights only'])
#     other=st.text_input('Optional clean-only checkpoint path','',help='Load a second checkpoint to compare both models on exactly the same degraded pixels.')
#     names_path=st.text_input('Optional class-name JSON path','',help='43 strings ordered by output ID, or object with keys 0 through 42. Defaults to Class 0…Class 42 because the supplied source has no named label mapping.')
#     inp=st.file_uploader('Cropped sign image',type=['png','jpg','jpeg','ppm'])
#     source=st.radio('Image input',['Uploaded file','Camera snapshot'],horizontal=True)
#     if source=='Camera snapshot':inp=st.camera_input('Photograph a tightly framed sign')
#     kind=st.selectbox('Degradation type',list(GRID),format_func=lambda k:TITLES[k],key='live_kind')
#     severity=st.select_slider('Selected severity',options=['Low','Medium','High'],value='Medium')
#     seed=st.number_input('Repeatable corruption seed',0,2147483647,42)
#     truth=st.selectbox('Known true class ID (optional)',[None]+list(range(43)),format_func=lambda n:'Unknown — do not score correctness' if n is None else f'Class {n}')
#     exists=Path(path).expanduser().is_file()
#     if not exists:st.warning('No trained checkpoint found at this path. Add your real best.keras or weights file. No predictions are generated until it is loaded.')
#     if st.button('Run visual demonstration',type='primary',disabled=not exists or inp is None):
#         try:
#             from visual_demo.core import preprocess_bytes,variants,predict,class_names,fingerprint,gradcam,feature_maps
#             p=Path(path).expanduser().resolve();stat=p.stat();model=cached_model(str(p),mode,stat.st_size,stat.st_mtime_ns)
#             names=class_names(names_path or None)
#             x=preprocess_bytes(inp.getvalue());batch=variants(x,kind,int(seed));probs=predict(model,batch)
#             other_probs=None;other_hash=None
#             if other:
#                 q=Path(other).expanduser().resolve();ss=q.stat();comparison=cached_model(str(q),mode,ss.st_size,ss.st_mtime_ns)
#                 other_probs=predict(comparison,batch);other_hash=fingerprint(q)
#             st.session_state['run']={'images':batch,'probs':probs,'other_probs':other_probs,'kind':kind,'seed':int(seed),'names':names,'truth':truth,'path':str(p),'mode':mode,'stat':(stat.st_size,stat.st_mtime_ns),'hash':fingerprint(p),'comparison_hash':other_hash}
#         except Exception as exc:st.error(f'Could not run the checkpoint: {exc}')
#     if 'run' in st.session_state:
#         run=st.session_state['run'];imgs=run['images'];pp=run['probs'];j=['Low','Medium','High'].index(severity)+1
#         st.caption(f"Last completed run: {TITLES[run['kind']]} · seed {run['seed']} · {Path(run['path']).name}. Change controls and click Run to recompute.")
#         fig=plots.live_grid(imgs,pp,run['kind'],run['names'],run['truth'],run['other_probs']);st.pyplot(fig)
#         st.download_button('Download prediction figure (300 dpi)',plots.png_bytes(fig),'prediction_comparison.png','image/png');plt.close(fig)
#         labels=run['names'];truth0=run['truth'];target=int(pp[0].argmax()) if truth0 is None else int(truth0)
#         st.write('Confidence for the same reference class across severity')
#         conf=pd.DataFrame({'Main model':pp[:,target]*100},index=['Clean','Low','Medium','High'])
#         if run['other_probs'] is not None:conf['Comparison model']=run['other_probs'][:,target]*100
#         st.line_chart(conf,y_label='Softmax score (%)');st.caption(f'Reference: {labels[target]}. Confidence is not accuracy or a calibrated probability of correctness.')
#         top=np.argsort(pp[j])[-5:][::-1]
#         st.dataframe(pd.DataFrame({'Selected input top 5':[labels[int(i)] for i in top],'Softmax (%)':pp[j,top]*100}),hide_index=True)
#         payload={'checkpoint_sha256':run['hash'],'comparison_sha256':run['comparison_hash'],'kind':run['kind'],'severity_settings':GRID[run['kind']],'seed':run['seed'],'true_class':truth0,'main_probabilities':pp.tolist(),'comparison_probabilities':None if run['other_probs'] is None else run['other_probs'].tolist(),'note':'Single-image illustration, not test-set accuracy.'}
#         st.download_button('Download prediction data',json.dumps(payload,indent=2),'prediction_data.json','application/json')
#         with st.expander('Grad-CAM and feature maps for the main model'):
#             st.caption('Grad-CAM highlights coarse regions contributing to a class score. The last activation is only 4 × 4; the enlarged overlay cannot resolve fine boundaries and does not prove correct reasoning.')
#             if st.button('Generate explanation'):
#                 try:
#                     from visual_demo.core import gradcam,feature_maps
#                     mm=cached_model(run['path'],run['mode'],*run['stat'])
#                     fig,axs=plt.subplots(1,2,figsize=(8,4))
#                     for ax,index,title in zip(axs,[0,j],['Clean',severity]):
#                         hm=gradcam(mm,imgs[index],target)
#                         ax.imshow(imgs[index],interpolation='nearest');ax.imshow(hm,cmap='inferno',alpha=.45,extent=(-.5,63.5,63.5,-.5),vmin=0,vmax=1)
#                         ax.set_title(f'{title} · {labels[target]}');ax.axis('off')
#                     st.pyplot(fig);st.download_button('Download Grad-CAM figure',plots.png_bytes(fig),'gradcam.png','image/png');plt.close(fig)
#                     act=feature_maps(mm,imgs[j],'conv_init_relu6');fig,axs=plt.subplots(2,4,figsize=(9,4.5))
#                     for index,ax in enumerate(axs.flat):ax.imshow(act[:,:,index],cmap='viridis');ax.set_title(f'Channel {index}');ax.axis('off')
#                     fig.suptitle('First 8 channels · conv_init_relu6 · channel-wise display scales')
#                     st.pyplot(fig);st.download_button('Download feature maps',plots.png_bytes(fig),'feature_maps.png','image/png');plt.close(fig)
#                 except Exception as exc:st.error(f'Explanation unavailable: {exc}')
# with architecture_tab:
#     fig=plots.architecture();st.pyplot(fig);plt.close(fig)
#     st.write('The exact proposed source architecture is included in src/models/proposed.py. The degradation module is copied unchanged from your upload. Decode → RGB → OpenCV INTER_AREA resize → divide by 255 matches the preprocessing module recovered from the nested Colab ZIP.')
#     st.write('Low light uses output = brightness × input^(1/gamma), with gamma 0.7/0.5/0.3 and brightness 0.90/0.75/0.60. Degradation is applied after resizing, matching the evaluation pipeline.')

"""Launch: python -m streamlit run app.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from visual_demo.results import (
    ROOT, GRID, TITLES, RUNS, LABELS,
    load_results, audit_results, rows
)
from visual_demo import plots
from visual_demo.core import class_names

st.set_page_config(
    page_title='TINA | Traffic sign robustness',
    layout='wide'
)

st.title('Traffic sign recognition under degraded conditions')
st.caption(
    'TINA thesis demonstration · TensorFlow/Keras · '
    '64 × 64 RGB · 43 classes'
)

results_tab, live_tab, architecture_tab = st.tabs([
    'Saved experiment results',
    'Live image demonstration',
    'How the model works'
])

data = load_results()

with results_tab:
    st.info(
        'These are archived metrics from your uploaded ZIP, '
        'not new inference results. No checkpoint is needed for this tab.'
    )

    try:
        audit_notes = audit_results(data)
    except AssertionError as exc:
        st.error(str(exc))
        st.stop()

    main = data[RUNS[0]]['evaluation']
    base = data[RUNS[1]]['evaluation']

    c = st.columns(4)

    c[0].metric(
        'Clean accuracy',
        f"{main['clean']['accuracy']:.2%}"
    )

    c[1].metric(
        'Mean degraded accuracy',
        f"{main['mean_degraded_accuracy']:.2%}",
        f"{100 * (main['mean_degraded_accuracy'] - base['mean_degraded_accuracy']):.2f} "
        'percentage points vs clean-only'
    )

    c[2].metric(
        'Robustness R',
        f"{main['robustness_score']:.3f}"
    )

    c[3].metric(
        'Archived test samples',
        f"{data[RUNS[0]]['num_test']:,}"
    )

    st.caption(
        'R = mean accuracy across 18 degraded conditions ÷ clean accuracy. '
        'Softmax confidence on one image is a different quantity.'
    )

    st.dataframe(
        pd.DataFrame(rows(data)),
        hide_index=True
    )

    kind = st.selectbox(
        'Inspect a degradation',
        list(GRID),
        format_func=lambda k: TITLES[k]
    )

    chart = {
        LABELS[i]: [
            100 * data[r]['evaluation']['per_condition'][f'{kind}/{s}']['accuracy']
            for s in GRID[kind]
        ]
        for i, r in enumerate(RUNS)
    }

    st.line_chart(
        pd.DataFrame(chart, index=['Low', 'Medium', 'High']),
        y_label='Accuracy (%)',
        x_label='Severity'
    )

    st.caption(f'Actual settings: {GRID[kind]}')

    fig = plots.comparison(data)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander('Result consistency and provenance'):
        st.write('\n\n'.join(audit_notes))
        st.write(
            'Checks confirm arithmetic consistency only. '
            'The original dataset, split manifest and trained checkpoints '
            'were not included, so archived experimental provenance cannot '
            'be independently reproduced here. '
            'The dummy smoke-test run is excluded.'
        )

    for name in [
        '01_architecture',
        '02_model_comparison',
        '03_robustness_curves',
        '04_condition_heatmap',
        '05_confusion_clean'
    ]:
        f = ROOT / 'thesis_figures' / f'{name}.png'

        if f.exists():
            st.download_button(
                f'Download {name}',
                f.read_bytes(),
                file_name=f.name,
                mime='image/png'
            )


@st.cache_resource
def cached_model(path, mode, size, mtime):
    from visual_demo.core import load_checkpoint
    return load_checkpoint(path, mode)


with live_tab:
    st.write(
        'Upload a tightly cropped traffic-sign image. '
        'Adjust the degradation and examine the model response. '
        'This classifier does not locate signs in full road scenes.'
    )

    path = st.text_input(
        'Main checkpoint path',
        str(ROOT / 'checkpoints/proposed_aware/best.keras')
    )

    mode = st.selectbox(
        'Checkpoint format',
        ['Full saved model', 'Proposed weights only']
    )

    other = st.text_input(
        'Optional clean-only checkpoint path',
        '',
        help=(
            'Load a second checkpoint to compare both models '
            'on exactly the same degraded pixels.'
        )
    )

    names_path = st.text_input(
        'Optional class-name JSON path',
        '',
        help=(
            'Leave blank to use the built-in GTSRB class names. '
            'An optional JSON override must contain 43 names ordered '
            'by output ID, or an object with keys 0 through 42.'
        )
    )

    inp = st.file_uploader(
        'Cropped sign image',
        type=['png', 'jpg', 'jpeg', 'ppm']
    )

    source = st.radio(
        'Image input',
        ['Uploaded file', 'Camera snapshot'],
        horizontal=True
    )

    if source == 'Camera snapshot':
        inp = st.camera_input('Photograph a tightly framed sign')

    kind = st.selectbox(
        'Degradation type',
        list(GRID),
        format_func=lambda k: TITLES[k],
        key='live_kind'
    )

    severity = st.select_slider(
        'Selected severity',
        options=['Low', 'Medium', 'High'],
        value='Medium'
    )

    seed = st.number_input(
        'Repeatable corruption seed',
        0,
        2147483647,
        42
    )

    # Use the same class names for the dropdown and predictions.
    try:
        true_class_labels = class_names(names_path or None)
    except Exception as exc:
        st.error(f'Could not load the class names: {exc}')
        st.stop()

    truth = st.selectbox(
        'Known true class ID (optional)',
        options=[None] + list(range(43)),
        format_func=lambda n: (
            'Unknown — do not score correctness'
            if n is None
            else true_class_labels[n]
        )
    )

    exists = Path(path).expanduser().is_file()

    if not exists:
        st.warning(
            'No trained checkpoint found at this path. '
            'Add your real best.keras or weights file. '
            'No predictions are generated until it is loaded.'
        )

    if st.button(
        'Run visual demonstration',
        type='primary',
        disabled=not exists or inp is None
    ):
        try:
            from visual_demo.core import (
                preprocess_bytes,
                variants,
                predict,
                fingerprint,
                gradcam,
                feature_maps
            )

            p = Path(path).expanduser().resolve()
            stat = p.stat()

            model = cached_model(
                str(p),
                mode,
                stat.st_size,
                stat.st_mtime_ns
            )

            names = true_class_labels

            x = preprocess_bytes(inp.getvalue())
            batch = variants(x, kind, int(seed))
            probs = predict(model, batch)

            other_probs = None
            other_hash = None

            if other:
                q = Path(other).expanduser().resolve()
                ss = q.stat()

                comparison = cached_model(
                    str(q),
                    mode,
                    ss.st_size,
                    ss.st_mtime_ns
                )

                other_probs = predict(comparison, batch)
                other_hash = fingerprint(q)

            st.session_state['run'] = {
                'images': batch,
                'probs': probs,
                'other_probs': other_probs,
                'kind': kind,
                'seed': int(seed),
                'names': names,
                'truth': truth,
                'path': str(p),
                'mode': mode,
                'stat': (stat.st_size, stat.st_mtime_ns),
                'hash': fingerprint(p),
                'comparison_hash': other_hash
            }

        except Exception as exc:
            st.error(f'Could not run the checkpoint: {exc}')

    if 'run' in st.session_state:
        run = st.session_state['run']
        imgs = run['images']
        pp = run['probs']
        j = ['Low', 'Medium', 'High'].index(severity) + 1

        st.caption(
            f"Last completed run: {TITLES[run['kind']]} · "
            f"seed {run['seed']} · {Path(run['path']).name}. "
            'Change controls and click Run to recompute.'
        )

        fig = plots.live_grid(
            imgs,
            pp,
            run['kind'],
            run['names'],
            run['truth'],
            run['other_probs']
        )

        st.pyplot(fig)

        st.download_button(
            'Download prediction figure (300 dpi)',
            plots.png_bytes(fig),
            'prediction_comparison.png',
            'image/png'
        )

        plt.close(fig)

        labels = run['names']
        truth0 = run['truth']
        target = int(pp[0].argmax()) if truth0 is None else int(truth0)

        st.write('Confidence for the same reference class across severity')

        conf = pd.DataFrame(
            {'Main model': pp[:, target] * 100},
            index=['Clean', 'Low', 'Medium', 'High']
        )

        if run['other_probs'] is not None:
            conf['Comparison model'] = (
                run['other_probs'][:, target] * 100
            )

        st.line_chart(
            conf,
            y_label='Softmax score (%)'
        )

        st.caption(
            f'Reference: {labels[target]}. '
            'Confidence is not accuracy or a calibrated probability '
            'of correctness.'
        )

        top = np.argsort(pp[j])[-5:][::-1]

        st.dataframe(
            pd.DataFrame({
                'Selected input top 5': [
                    labels[int(i)] for i in top
                ],
                'Softmax (%)': pp[j, top] * 100
            }),
            hide_index=True
        )

        payload = {
            'checkpoint_sha256': run['hash'],
            'comparison_sha256': run['comparison_hash'],
            'kind': run['kind'],
            'severity_settings': GRID[run['kind']],
            'seed': run['seed'],
            'true_class': truth0,
            'main_probabilities': pp.tolist(),
            'comparison_probabilities': (
                None
                if run['other_probs'] is None
                else run['other_probs'].tolist()
            ),
            'note': 'Single-image illustration, not test-set accuracy.'
        }

        st.download_button(
            'Download prediction data',
            json.dumps(payload, indent=2),
            'prediction_data.json',
            'application/json'
        )

        with st.expander('Grad-CAM and feature maps for the main model'):
            st.caption(
                'Grad-CAM highlights coarse regions contributing to a '
                'class score. The last activation is only 4 × 4; '
                'the enlarged overlay cannot resolve fine boundaries '
                'and does not prove correct reasoning.'
            )

            if st.button('Generate explanation'):
                try:
                    from visual_demo.core import gradcam, feature_maps

                    mm = cached_model(
                        run['path'],
                        run['mode'],
                        *run['stat']
                    )

                    fig, axs = plt.subplots(
                        1,
                        2,
                        figsize=(8, 4)
                    )

                    for ax, index, title in zip(
                        axs,
                        [0, j],
                        ['Clean', severity]
                    ):
                        hm = gradcam(
                            mm,
                            imgs[index],
                            target
                        )

                        ax.imshow(
                            imgs[index],
                            interpolation='nearest'
                        )

                        ax.imshow(
                            hm,
                            cmap='inferno',
                            alpha=.45,
                            extent=(-.5, 63.5, 63.5, -.5),
                            vmin=0,
                            vmax=1
                        )

                        ax.set_title(
                            f'{title} · {labels[target]}'
                        )

                        ax.axis('off')

                    st.pyplot(fig)

                    st.download_button(
                        'Download Grad-CAM figure',
                        plots.png_bytes(fig),
                        'gradcam.png',
                        'image/png'
                    )

                    plt.close(fig)

                    act = feature_maps(
                        mm,
                        imgs[j],
                        'conv_init_relu6'
                    )

                    fig, axs = plt.subplots(
                        2,
                        4,
                        figsize=(9, 4.5)
                    )

                    for index, ax in enumerate(axs.flat):
                        ax.imshow(
                            act[:, :, index],
                            cmap='viridis'
                        )
                        ax.set_title(f'Channel {index}')
                        ax.axis('off')

                    fig.suptitle(
                        'First 8 channels · conv_init_relu6 · '
                        'channel-wise display scales'
                    )

                    st.pyplot(fig)

                    st.download_button(
                        'Download feature maps',
                        plots.png_bytes(fig),
                        'feature_maps.png',
                        'image/png'
                    )

                    plt.close(fig)

                except Exception as exc:
                    st.error(f'Explanation unavailable: {exc}')


with architecture_tab:
    fig = plots.architecture()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'The exact proposed source architecture is included in '
        'src/models/proposed.py. The degradation module is copied '
        'unchanged from your upload. Decode → RGB → OpenCV INTER_AREA '
        'resize → divide by 255 matches the preprocessing module '
        'recovered from the nested Colab ZIP.'
    )

    st.write(
        'Low light uses output = brightness × input^(1/gamma), '
        'with gamma 0.7/0.5/0.3 and brightness 0.90/0.75/0.60. '
        'Degradation is applied after resizing, matching the '
        'evaluation pipeline.'
    )