"""Render the report's editable vector overview and high-resolution PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(19, 11))
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 19), ylim=(0, 11))
ax.axis('off')
blue, red, purple, gold = '#dceaff', '#f9dedb', '#e8def3', '#fff0cc'
ink, muted = '#263344', '#607080'

def text(x, y, label, size=11, weight='normal', color=ink, ha='center'):
    ax.text(x, y, label, ha=ha, va='center', fontsize=size, weight=weight, color=color, linespacing=1.5)

def box(x, y, w, h, label, color='white', size=11, dashed=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.035,rounding_size=0.10', linewidth=1.1,
        edgecolor='#9aa6b2', facecolor=color, linestyle=(0,(4,3)) if dashed else '-'))
    if label:
        text(x, y, label, size)

def arrow(points, color=muted, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.5, ls='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
        mutation_scale=14, linewidth=1.5, color=color, linestyle='--' if dashed else '-'))

text(.45, 10.6, 'TQSI  |  Overall implemented architecture', 22, 'bold', ha='left')
text(.45, 10.15, 'Frozen SAM spatial features + trainable quantum conditioning + semantic image refinement', 12, color=muted, ha='left')
for x, label, color in [(11.8, 'Frozen', blue), (14.2, 'Trainable', red), (16.8, 'Quantum', purple)]:
    box(x, 9.55, 2.0, .42, label, color, 10)

# Light grouped panels echo the reference without importing its unrelated VAE/router.
box(3.35, 5.15, 5.8, 8.25, '', '#f8fafc', dashed=True)
box(9.65, 5.15, 5.6, 8.25, '', '#fbf8fe', dashed=True)
box(15.8, 5.15, 5.3, 8.25, '', '#fffaf1', dashed=True)
text(3.35, 8.92, '01  SPATIAL SEGMENTATION', 12, 'bold')
text(9.65, 8.92, '02  QUANTUM CONDITIONING', 12, 'bold')
text(15.8, 8.92, '03  TRAINING & CONTINUAL FLOW', 12, 'bold')

box(3.2, 1.65, 3.75, .72, 'RGB tile\nB × 3 × 512 × 512', size=11)
box(3.2, 2.95, 3.75, 1.02, 'Frozen SAM ViT-B encoder\nResize to 1024 × 1024; no_grad\nFeatures: B × 256 × 64 × 64', blue, 10.5)
box(3.2, 4.25, 3.75, .8, 'Spatial residual adapter\n1×1: 256 → 32 → 256 + skip', red)
box(3.2, 6.15, 3.75, 1.48, 'FiLM semantic decoder\n3×3 stem → scale / shift → residual body\nUpsample + RGB branch → concatenate\n3×3 refinement → 1×1 classifier', red, 10)
box(3.2, 8.05, 3.75, .8, 'Predicted building mask\nLogits → calibrated threshold', gold)
arrow([(3.2,2.01),(3.2,2.44)])
arrow([(3.2,3.46),(3.2,3.85)])
arrow([(3.2,4.65),(3.2,5.41)])
text(3.2,5.03,'Dense spatial features',9,color=muted)
arrow([(3.2,6.89),(3.2,7.65)])
text(3.2,7.25,'B × 1 × 512 × 512 logits',9,color=muted)
arrow([(1.3,1.65),(.83,1.65),(.83,6.15),(1.3,6.15)])
ax.text(.64,4.5,'RGB detail branch',rotation=90,ha='center',va='center',fontsize=9,color=muted)

box(9.65, 2.1, 4.65, 1.0, 'Global projection\nAverage pool → Linear 256→256\nLayerNorm → L2 normalization', red, 10.5)
arrow([(5.1,4.25),(6.55,4.25),(6.55,2.1),(7.3,2.1)])
box(9.65, 3.6, 4.65, .83, 'Amplitude encoding\n256 real features → 8-qubit state', purple)
arrow([(9.65,2.6),(9.65,3.18)])
box(9.65, 5.35, 4.65, 1.95, '', purple)
text(9.65,6.04,'Variational circuit  × 6 layers',12,'bold')
text(9.65,5.65,'Rot(3 angles) on every wire q0 … q7',10)
text(9.65,5.28,'CNOT: 0→1, 1→2, 2→3, 3→4',10)
text(9.65,4.97,'then 4→5, 5→6, 6→7, 7→0',10)
text(9.65,4.63,'144 angles total · exact range-1 ring sequence',9)
arrow([(9.65,4.015),(9.65,4.375)])
box(9.65, 7.07, 4.65, .78, 'Pauli readout: ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit\n24 real values → Linear → FiLM scale / shift', purple, 10)
arrow([(9.65,6.325),(9.65,6.68)])
arrow([(7.3,7.07),(6.55,7.07),(6.55,6.15),(5.1,6.15)],'#8766a9')
text(9.65,8.07,'Exact state-vector simulation\ndefault.qubit · shots=None · Torch backprop',10,color=muted)

box(15.8, 8.0, 4.5, .85, 'Ground-truth masks + predicted logits\nFocal-weighted BCE + Tversky', gold, 10.5)
box(15.8, 6.55, 4.5, .9, 'Continual terms (later tasks)\nLabeled replay + logit distillation\nReference-state stability / separation*', gold, 10)
arrow([(15.8,7.575),(15.8,7.0)])
box(15.8, 5.05, 4.5, .95, 'Backpropagation → masked AdamW\nQuantum: 72 active angles / task (2 tasks)\nShared classical heads update normally', red, 10)
arrow([(15.8,6.1),(15.8,5.525)])
box(15.8, 3.45, 4.5, 1.0, 'Epoch end\nFull validation: calibrate threshold\nFull validation again: metrics + best model', gold, 10)
arrow([(15.8,4.575),(15.8,3.95)])
box(15.8, 1.95, 4.5, .85, 'Task boundary\nBest weights → replay memory → next task\nEvaluate all seen tasks', gold, 10)
arrow([(15.8,2.95),(15.8,2.375)])

text(.5,.66,'Shown: width 256 / rank 32 / 8 qubits / 6 layers  •  2,313,106 trainable parameters  •  93,735,472 frozen parameters',11,'bold',ha='left')
text(.5,.26,'Remote configuration requires confirmation. *Shared-unitary separation is invariant for fixed reference inputs. SAM prompt/mask decoder is unused in this path.',9,color=muted,ha='left')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
for extension in ('svg','png'):
    fig.savefig(OUT / f'tqsi_overall_architecture.{extension}',dpi=190,facecolor='white')
plt.close(fig)
print(OUT / 'tqsi_overall_architecture.png')
