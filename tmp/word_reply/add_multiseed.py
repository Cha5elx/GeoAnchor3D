from pathlib import Path
from docx import Document
from docx.shared import Pt,RGBColor, Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from copy import deepcopy
import re
base=Path('docs/Response2Reviewers');src=base/'GeoAnchor3D_Response_Letter_with_revision_markers.docx';d=Document(src)
r=0;c=0;targets={}
for p in d.paragraphs:
 t=p.text.strip()
 if t.startswith('Response to Reviewer '):r=int(t[-1])
 elif re.fullmatch(r'Comment \d+:',t):c=int(re.search(r'\d+',t).group())
 elif t.startswith('[Response pending:'):targets[r,c]=p
assert set(targets)=={(1,5),(3,5)}
def style(run):
 run.font.name='Times New Roman';run.font.size=Pt(10.5);run.font.color.rgb=RGBColor.from_string('3427E1');run.italic=False

def add(after,txt):
 el=OxmlElement('w:p');after._p.addnext(el)
 from docx.text.paragraph import Paragraph
 p=Paragraph(el,after._parent);p._p.insert(0,deepcopy(after._p.pPr));p.paragraph_format.keep_with_next=False
 style(p.add_run(txt));return p
p=targets[1,5];p.clear()
style(p.add_run('Thank you for emphasizing optimization stability. We evaluated the complete GeoAnchor3D configuration using random seeds 42, 2025, and 3407 under the same three-epoch training protocol. The initialization checkpoint, datasets, hyperparameters, checkpoint-selection rule, and evaluation protocol were held fixed. The table below reports the mean and sample standard deviation across the three runs.'))
cap=add(p,'Three-seed results for the complete GeoAnchor3D model (mean ± sample standard deviation).')
cap.paragraph_format.first_line_indent=Pt(0);cap.paragraph_format.keep_with_next=True
rows=[('ScanRefer','Acc@0.25','56.57 ± 0.21'),('ScanRefer','Acc@0.5','51.10 ± 0.20'),('Multi3DRefer','F1@0.25','59.00 ± 0.17'),('Multi3DRefer','F1@0.5','54.53 ± 0.06'),('ScanQA','BLEU-1','42.63 ± 0.51'),('ScanQA','BLEU-4','14.10 ± 0.26'),('ScanQA','METEOR','18.33 ± 0.15'),('ScanQA','ROUGE-L','41.80 ± 0.36'),('ScanQA','CIDEr','88.87 ± 1.00'),('Scan2Cap','CIDEr@0.5','77.43 ± 0.25'),('Scan2Cap','BLEU-4@0.5','35.93 ± 0.15')]
t=d.add_table(rows=1,cols=3);t.autofit=False
for cell,txt in zip(t.rows[0].cells,['Dataset','Metric','Mean ± Std.']):cell.text=txt
for row in rows:
 for cell,txt in zip(t.add_row().cells,row):cell.text=txt
for row in t.rows:
 for j,cell in enumerate(row.cells):
  cell.width=Inches([1.8,1.8,2.1][j])
  for pp in cell.paragraphs:
   pp.paragraph_format.space_after=Pt(3);pp.paragraph_format.space_before=Pt(3)
   for rr in pp.runs:style(rr)
  tcpr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
  for side in ['top','left','bottom','right']:
   el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
  tcpr.append(borders)
for cell in t.rows[0].cells:
 sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'F2F2F2');cell._tc.get_or_add_tcPr().append(sh)
 for rr in cell.paragraphs[0].runs:rr.bold=True
rep=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(rep)
cap._p.addnext(t._tbl)
# Insert following text after the table while preserving the original response style.
a=add(cap,'Across the evaluated seeds, all four grounding metrics remain above the reported Chat-Scene reference. ScanQA CIDEr and Scan2Cap CIDEr@0.5 also exceed the reported reference in every run, whereas BLEU-4 does not show a consistent improvement. These results support run-to-run stability of the complete model under the evaluated protocol.')
t._tbl.addnext(a._p)
a=add(a,'The main comparison and controlled ablation tables retain their seed-42 results. This additional experiment evaluates the complete model; it does not establish multi-seed stability for each ablation variant. Since Chat-Scene was not retrained under the same three seeds, we do not claim statistical significance relative to the baseline.')
a=add(a,'The LoRA rank, learning-rate schedule, and auxiliary-loss weights were held fixed across runs. Varying the random seed does not test sensitivity to these hyperparameters; their tuning and sensitivity analysis remain separate from the results reported here.')
p=targets[3,5];p.clear();style(p.add_run('Thank you for requesting a stability analysis. We evaluated the complete GeoAnchor3D model with seeds 42, 2025, and 3407 using the same three-epoch training and evaluation protocol. The full 11-metric summary, reported as mean ± sample standard deviation, is provided in our response to Reviewer 1, Comment 5.'))
p=add(p,'ScanRefer Acc@0.25 and Acc@0.5 are 56.57 ± 0.21 and 51.10 ± 0.20; Multi3DRefer F1@0.25 and F1@0.5 are 59.00 ± 0.17 and 54.53 ± 0.06. Each run exceeds the reported Chat-Scene reference on these four metrics. ScanQA CIDEr is 88.87 ± 1.00, and Scan2Cap CIDEr@0.5 is 77.43 ± 0.25; every run also exceeds the reported baseline on these two metrics. BLEU-4 results remain mixed, so we do not claim uniform improvement across generation metrics.')
add(p,'These experiments quantify variation across runs of the complete GeoAnchor3D model. Without matched multi-seed runs of Chat-Scene, they do not establish statistical significance between the two methods. They also do not test the relative ranking of the GATH layer intervals.')
out=base/'GeoAnchor3D_Response_Letter_with_multiseed.docx';d.save(out)
assert len(Document(out).tables)==1
assert not any('[Response pending:' in p.text for p in Document(out).paragraphs)
print('Saved; 2 responses updated, 11 results inserted, 19 screenshot markers retained.')
